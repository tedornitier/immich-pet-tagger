"""Poller: incremental classification using a local CLIP model.
No DB access. Embeddings computed from thumbnails via the Immich HTTP API."""

import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import classifier as clf_mod
import data
import embedder as emb
import immich as imm

log = logging.getLogger("poller")

THRESHOLD = float(os.environ.get("THRESHOLD", 0.8))

_count_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Date range helpers
# ---------------------------------------------------------------------------

def _advance_ms(ts: str) -> str:
    """Bump an ISO timestamp 1ms past the latest processed asset before saving it as
    the next cursor. Immich's createdAfter/takenAfter filters are inclusive, so saving
    the asset's own timestamp verbatim would match that same asset again next cycle."""
    dt = datetime.fromisoformat(ts.replace("Z", "+00:00")) + timedelta(milliseconds=1)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def parse_date(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return date.fromisoformat(s[:10])
    except (ValueError, TypeError):
        return None


def asset_in_range(time_str: str, since: str | None, until: str | None) -> bool:
    d = parse_date(time_str)
    if d is None:
        return True
    if since and d < date.fromisoformat(since):
        return False
    if until and d > date.fromisoformat(until):
        return False
    return True


# ---------------------------------------------------------------------------
# Ref migration
# ---------------------------------------------------------------------------

def migrate_ref_bboxes(data_dir: Path) -> None:
    """One-time migration: fill in missing bbox and face_id fields on old-format refs."""
    config = data.load_config(data_dir)
    bbox_resolved = 0
    bbox_unresolvable = 0
    face_recovered = 0
    for pet_name, cfg in config.items():
        folder_key = cfg.get("person_id") or pet_name
        person_id = cfg.get("person_id")
        refs = data.load_pet_refs(folder_key, data_dir)
        changed = False
        for ref in refs:
            if not ref.get("bbox"):
                bbox = emb.resolve_bbox(ref["asset_id"])
                if bbox:
                    ref["bbox"] = bbox
                    changed = True
                    bbox_resolved += 1
                else:
                    bbox_unresolvable += 1
            if not ref.get("face_id") and person_id:
                face_id = imm.fetch_face_id_for_person(ref["asset_id"], person_id)
                if face_id:
                    ref["face_id"] = face_id
                    changed = True
                    face_recovered += 1
        if changed:
            data.save_pet_refs(folder_key, refs, data_dir)
    parts = []
    if bbox_resolved or bbox_unresolvable:
        parts.append(f"bbox: {bbox_resolved} resolved, {bbox_unresolvable} unresolvable")
    if face_recovered:
        parts.append(f"face_id: {face_recovered} recovered")
    if parts:
        log.info(f"Ref migration: {', '.join(parts)}")


# ---------------------------------------------------------------------------
# Main poll cycle
# ---------------------------------------------------------------------------

def run_poll_cycle(data_dir: str, on_date=None, cancel=None, low_conf_out=None, live_counts: dict | None = None, manual: bool = False, scan_until: str | None = None, scan_since: str | None = None) -> None:
    log.info(f"Poll cycle | threshold={THRESHOLD} manual={manual}")
    dd = Path(data_dir)
    now = datetime.now(timezone.utc).isoformat()
    data.write_poll_status(dd, {"status": "running", "started_at": now})

    counts = live_counts if live_counts is not None else {}
    for k in ("added", "low_confidence", "full_image", "unknown", "out_of_range", "already_tagged", "failed", "no_thumb"):
        counts[k] = 0
    try:
        _run_poll_cycle(dd, counts, on_date, cancel, low_conf_out, manual, scan_until, scan_since)
    except Exception as e:
        data.write_poll_status(dd, {"status": "error", "ran_at": datetime.now(timezone.utc).isoformat(), "error": str(e), "counts": counts})
        raise
    else:
        data.write_poll_status(dd, {"status": "idle", "ran_at": datetime.now(timezone.utc).isoformat(), "counts": counts})


def _run_poll_cycle(dd: Path, counts: dict, on_date=None, cancel=None, low_conf_out=None, manual: bool = False, scan_until: str | None = None, scan_since: str | None = None) -> None:
    config = data.load_config(dd)
    if not config:
        log.warning("config.json empty or missing, no pets configured yet.")
        return

    all_pet_names = list(config.keys())
    all_refs = {name: data.load_pet_refs(config[name].get("person_id") or name, dd) for name in all_pet_names}

    pet_names = [n for n in all_pet_names if all_refs.get(n)]
    refs_per_pet = {n: all_refs[n] for n in pet_names}
    skipped = [n for n in all_pet_names if n not in pet_names]

    if skipped:
        log.warning(f"Skipping pets with no refs: {skipped}")
    if not pet_names:
        log.warning("No pets with reference assets, enroll pets via the UI first.")
        return

    log.info(f"Pets: {', '.join(f'{n}({len(refs_per_pet[n])} refs)' for n in pet_names)}")

    negative_ids = data.load_negative_ids(dd)
    if negative_ids:
        log.info(f"Loaded {len(negative_ids)} negative samples")

    result = clf_mod.build_classifier(pet_names, refs_per_pet, negative_ids)
    if result is None:
        return
    names, clf, scaler = result

    last_ts = scan_since if manual else data.load_last_timestamp(dd)
    log.info(f"Fetching assets taken after: {last_ts}")

    t0 = time.time()
    if manual:
        taken_before = (scan_until + "T23:59:59.999Z") if scan_until else None
        assets = imm.fetch_assets_taken_after(last_ts, taken_before)
    else:
        assets = imm.fetch_assets_created_after(last_ts)
    log.info(f"Fetched {len(assets)} assets in {time.time()-t0:.1f}s")

    if not assets:
        log.info("No new assets.")
        if not manual:
            data.save_last_timestamp(datetime.now(timezone.utc).isoformat(), dd)
        return

    latest_ts = max((ts for _, ts in assets), default=last_ts)

    def process_asset(aid: str, time_str: str) -> None:
        if cancel and cancel.is_set():
            return

        if on_date:
            on_date(time_str[:10])

        img = emb.fetch_thumbnail(aid)
        if img is None:
            with _count_lock:
                counts["no_thumb"] += 1
            return
        detected = emb.crop_animals(img)
        if not detected:
            # YOLO found no animal even at YOLO_CONF, so fall back to the whole
            # image: it still finds pets the detector missed outright. But this
            # embedding describes the entire scene rather than an animal, and
            # the classifier always returns a best-matching class regardless of
            # image content, so a match here is far weaker evidence than one
            # backed by a real detection. bbox_norm stays None to mark that, and
            # the loop below never auto-tags such a match.
            crops = [(None, img)]
        else:
            crops = detected
            if len(detected) > 1:
                log.info(f"YOLO detected {len(detected)} animals in {aid} ({time_str[:10]})")
        vecs = [(bbox_norm, emb.embed_image(crop)) for bbox_norm, crop in crops]

        # Populate the crop cache so borderline and suggestions can reuse this
        # work without re-fetching and re-embedding. Only real animal crops are
        # stored; an empty list marks "no animal detected".
        emb.store_crops(aid, [(b, v) for b, v in vecs if b is not None and v is not None])

        existing_persons: set | None = None
        tagged_in_photo: set[str] = set()

        for bbox_norm, vec in vecs:
            if vec is None:
                continue

            # No detection behind this crop: it is the whole photo (see above).
            full_image = bbox_norm is None

            pet_name, prob = clf_mod.classify(vec, names, clf, scaler)

            if pet_name == "unknown":
                with _count_lock:
                    counts["unknown"] += 1
                continue

            # A whole-image match goes to review however confident it looks, since
            # the score reflects the whole scene and not an identified animal.
            if full_image or prob < THRESHOLD:
                with _count_lock:
                    if full_image and prob >= THRESHOLD:
                        counts["full_image"] += 1
                    else:
                        counts["low_confidence"] += 1
                if low_conf_out is not None:
                    low_conf_out.append({"asset_id": aid, "pet_name": pet_name, "prob": prob,
                                         "date": time_str[:10],
                                         "bbox": None if full_image else list(bbox_norm),
                                         "full_image": full_image})
                continue

            cfg = config.get(pet_name, {})
            if not asset_in_range(time_str, cfg.get("since"), cfg.get("until")):
                with _count_lock:
                    counts["out_of_range"] += 1
                continue

            person_id = cfg.get("person_id")
            if not person_id:
                log.warning(f"Pet '{pet_name}' has no person_id in config.")
                continue

            if person_id in tagged_in_photo:
                with _count_lock:
                    counts["already_tagged"] += 1
                continue

            if existing_persons is None:
                existing_persons = imm.fetch_asset_face_person_ids(aid)

            if person_id in existing_persons:
                with _count_lock:
                    counts["already_tagged"] += 1
                continue

            log.info(f"{imm.IMMICH_URL}/search/photos/{aid} -> {pet_name} ({prob:.3f}) | {time_str[:10]}")

            face_id = imm.post_face_sync(aid, person_id, bbox_norm, img.size)
            tagged_in_photo.add(person_id)
            with _count_lock:
                if face_id:
                    counts["added"] += 1
                else:
                    counts["failed"] += 1

    import detector as _det
    emb.reset_batch_stats()
    with _det._yolo_stats_lock:
        _det.yolo_batch_total = _det.yolo_batch_count = 0

    log.info(f"Processing {len(assets)} assets with {emb.SCAN_WORKERS} workers")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=emb.SCAN_WORKERS) as executor:
        futures = {executor.submit(process_asset, aid, ts): aid for aid, ts in assets}
        for future in as_completed(futures):
            if cancel and cancel.is_set():
                executor.shutdown(wait=False, cancel_futures=True)
                log.info("Scan cancelled.")
                return
            try:
                future.result()
            except Exception as e:
                log.warning(f"Asset {futures[future]} failed: {e}")

    elapsed = time.time() - t0
    clip_avg = emb.get_avg_batch_size()
    with _det._yolo_stats_lock:
        yolo_avg = _det.yolo_batch_total / _det.yolo_batch_count if _det.yolo_batch_count else 0
    log.info(
        f"STATS | assets={len(assets)} elapsed={elapsed:.1f}s "
        f"throughput={len(assets)/elapsed:.1f}/s "
        f"yolo_batch={yolo_avg:.1f} clip_batch={clip_avg:.1f} "
        f"counts={counts}"
    )

    emb.save_embed_cache()

    if not manual:
        next_ts = _advance_ms(latest_ts)
        data.save_last_timestamp(next_ts, dd)
        log.info(f"Saved timestamp: {next_ts}")
