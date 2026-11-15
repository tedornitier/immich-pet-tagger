"""API routes for the enrollment UI.
All Immich communication happens here; the browser never touches Immich directly."""

import asyncio
import json
import logging
import os
import shutil
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import httpx
import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel

import io

import data
import detector as det
import embedder as emb
import immich as imm
import state
from embedder import embed_asset
from inference import describe_model_failure, inference_session, model_error_count

log = logging.getLogger("api")

router = APIRouter(prefix="/api")

IMMICH_EXTERNAL_URL = os.environ.get("IMMICH_EXTERNAL_URL", "http://localhost:2283")
DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
PETS_DIR = DATA_DIR / "pets"
LONG_REQUEST_TIMEOUT = int(os.environ.get("LONG_REQUEST_TIMEOUT", 120))
KEEPALIVE_INTERVAL = 15


async def _streaming_json(coro):
    """Stream JSON with periodic keepalive bytes while CPU-heavy work runs.
    Browsers drop idle connections after ~90s with no response bytes."""
    async def generate():
        task = asyncio.create_task(coro)
        while not task.done():
            yield b" \n"
            await asyncio.wait([task], timeout=KEEPALIVE_INTERVAL)  # never raises, even if the task did
        try:
            result = await task
        except HTTPException as e:
            # The 200 and the keepalive bytes are already sent, so the status code can no
            # longer change. app.js api() turns this body into a thrown error.
            result = {"stream_error": e.detail}
        yield json.dumps(result).encode()

    return StreamingResponse(generate(), media_type="application/json")


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class PetCreate(BaseModel):
    name: str
    since: Optional[str] = None
    until: Optional[str] = None
    description: str


class PetUpdate(BaseModel):
    name: Optional[str] = None
    since: Optional[str] = None
    until: Optional[str] = None
    description: Optional[str] = None


class PetAssets(BaseModel):
    asset_ids: list[str]


class CropRef(BaseModel):
    asset_id: str
    crop_idx: Optional[int] = None
    bbox: Optional[list[float]] = None


class NegativesBody(BaseModel):
    asset_ids: list[str] = []  # whole-photo entries ("Add manually", older clients)
    entries: list[CropRef] = []  # one entry per selected crop, bbox None means the whole photo


class PetCropAssets(BaseModel):
    asset_ids: Optional[list[str]] = None  # backwards compat
    assets: Optional[list[CropRef]] = None  # crop-centric format


_VERSION_FILE = Path(__file__).parent / "VERSION"

@router.get("/version")
async def get_version():
    try:
        return {"version": _VERSION_FILE.read_text().strip()}
    except FileNotFoundError:
        return {"version": "unknown"}


@router.get("/config")
async def get_config():
    return {
        "immich_external_url": IMMICH_EXTERNAL_URL,
        "models_ready": det.is_yolo_ready() and emb.is_clip_ready(),
        "models_error": det.get_yolo_error() or emb.get_clip_error(),
    }


@router.get("/settings")
async def get_settings():
    """Env-var-configured defaults (THRESHOLD, THRESHOLD_FALLBACK, YOLO_CONF). Read-only:
    production config lives in docker-compose.yml only. Used to prefill the benchmark
    tool's per-run threshold overrides, which are never persisted, they only apply to
    that one run."""
    from poller import THRESHOLD, THRESHOLD_FALLBACK
    return {"threshold": THRESHOLD, "threshold_fallback": THRESHOLD_FALLBACK, "yolo_conf": det.YOLO_CONF}


def _slim_asset(a: dict) -> dict:
    return {"id": a["id"], "thumb": f"/api/crop/{a['id']}", "date": a.get("localDateTime", "")[:10], "filename": a.get("originalFileName", "")}


def _pet_image_search_params(pet_cfg: dict) -> dict:
    """Search filter for images within the pet's since/until date range."""
    since, until = pet_cfg.get("since"), pet_cfg.get("until")
    return imm.search_params(
        asset_type="IMAGE",
        taken_after=since + "T00:00:00.000Z" if since else None,
        taken_before=until + "T23:59:59.999Z" if until else None,
    )


async def _visual_search(
    client: httpx.AsyncClient,
    ref_ids: list[str],
    pet_cfg: dict,
    exclude: set[str],
    sample: int = 8,
    per_ref_limit: int = 50,
) -> list[dict]:
    """Query Immich smart search using ref asset IDs instead of text.
    Runs all ref queries in parallel and returns deduplicated candidates."""
    if len(ref_ids) > sample:
        step = len(ref_ids) / sample
        sampled = [ref_ids[int(i * step)] for i in range(sample)]
    else:
        sampled = ref_ids

    base: dict = {"size": per_ref_limit, **_pet_image_search_params(pet_cfg)}

    async def fetch_one(rid: str) -> list[dict] | None:
        """Items for one ref query, or None if the query failed (an empty list is a real result)."""
        try:
            resp = await client.post(
                f"{imm.IMMICH_URL}/api/search/smart",
                headers=imm.headers(),
                json={**base, "queryAssetId": rid},
            )
            if resp.status_code == 200:
                return resp.json().get("assets", {}).get("items", [])
            log.warning(f"visual search for ref {rid}: HTTP {resp.status_code}: {resp.text[:200]}")
        except Exception as e:
            log.warning(f"visual search for ref {rid} failed: {e}")
        return None

    results = [r for r in await asyncio.gather(*[fetch_one(rid) for rid in sampled]) if r is not None]
    if sampled and not results:
        raise HTTPException(status_code=502, detail="Immich smart search failed for every reference photo, see the pet tagger logs")
    seen: set[str] = set()
    candidates: list[dict] = []
    for items in results:
        for a in items:
            aid = a.get("id")
            if aid and aid not in exclude and aid not in seen:
                seen.add(aid)
                candidates.append(a)
    return await asyncio.to_thread(imm.drop_foreign_assets, candidates, "visual search")


# ---------------------------------------------------------------------------
# Pets
# ---------------------------------------------------------------------------

@router.get("/pets")
async def list_pets():
    config = data.load_config(DATA_DIR)
    return {"pets": [
        {"name": name, "person_id": cfg.get("person_id"), "since": cfg.get("since"),
         "until": cfg.get("until"), "description": cfg.get("description"),
         "ref_count": len(data.load_pet_refs(cfg.get("person_id") or name, DATA_DIR))}
        for name, cfg in config.items()
    ]}


@router.post("/pets")
async def create_pet(pet: PetCreate):
    config = data.load_config(DATA_DIR)
    name = pet.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Name cannot be empty")
    if name.lower() in {k.lower() for k in config}:
        raise HTTPException(status_code=409, detail=f"Pet '{name}' already exists")

    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(f"{imm.IMMICH_URL}/api/people", headers=imm.headers(), json={"name": name})
    if resp.status_code not in (200, 201):
        raise HTTPException(status_code=resp.status_code, detail=f"Immich error: {resp.text}")

    person_id = resp.json().get("id")
    config[name] = {"person_id": person_id, "since": pet.since, "until": pet.until, "description": pet.description}
    data.save_config(config, DATA_DIR)
    (PETS_DIR / person_id).mkdir(parents=True, exist_ok=True)
    log.info(f"Created pet '{name}' with person_id={person_id}")
    return {"name": name, "person_id": person_id}


@router.patch("/pets/{name}")
async def update_pet(name: str, update: PetUpdate):
    config = data.load_config(DATA_DIR)
    if name not in config:
        raise HTTPException(status_code=404, detail=f"Pet '{name}' not found")

    new_name = update.name.strip() if update.name else None
    if new_name and new_name != name:
        if new_name.lower() in {k.lower() for k in config if k != name}:
            raise HTTPException(status_code=409, detail=f"Pet '{new_name}' already exists")
        person_id = config[name].get("person_id")
        if person_id:
            async with httpx.AsyncClient(timeout=15) as client:
                url = f"{imm.IMMICH_URL}/api/people/{person_id}"
                # PATCH replaces the PUT deprecated in Immich v3. Immich v2 only has PUT.
                resp = await client.patch(url, headers=imm.headers(), json={"name": new_name})
                if resp.status_code in (404, 405):
                    resp = await client.put(url, headers=imm.headers(), json={"name": new_name})
                if resp.status_code != 200:
                    log.warning(f"Failed to rename Immich person {person_id} to '{new_name}': {resp.status_code} {resp.text[:200]}")
        config[new_name] = config.pop(name)
        name = new_name

    if "since" in update.model_fields_set:
        config[name]["since"] = update.since
    if "until" in update.model_fields_set:
        config[name]["until"] = update.until
    if "description" in update.model_fields_set:
        config[name]["description"] = update.description
    data.save_config(config, DATA_DIR)
    log.info(f"Updated pet '{name}'")
    return {"ok": True}


@router.post("/pets/{name}/reset-immich")
async def reset_pet_immich(name: str):
    """Delete the Immich person for this pet (removing all face tags), create a fresh one,
    and preserve the local refs. face_ids in refs are cleared since the old person is gone."""
    config = data.load_config(DATA_DIR)
    if name not in config:
        raise HTTPException(status_code=404, detail=f"Pet '{name}' not found")

    old_person_id = config[name].get("person_id")
    old_refs = data.load_pet_refs(old_person_id, DATA_DIR) if old_person_id else []

    # Delete old Immich person. 404 means it was already removed manually, which is fine.
    if old_person_id:
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.delete(f"{imm.IMMICH_URL}/api/people/{old_person_id}", headers=imm.headers())
                if resp.status_code in (200, 204):
                    await _untag_pet_refs(client, config, name, old_refs)
        except httpx.ConnectError:
            raise HTTPException(status_code=503, detail="Cannot reach Immich. Is it running?")
        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail="Immich did not respond in time.")
        if resp.status_code == 404:
            log.warning(f"Immich person {old_person_id} for pet '{name}' not found, treating as already deleted")
        elif resp.status_code not in (200, 204):
            raise HTTPException(status_code=resp.status_code, detail=f"Immich error deleting person: {resp.text}")
        else:
            log.info(f"Deleted Immich person {old_person_id} for pet '{name}' (reset)")

    # Create new Immich person. If this fails, the old person is already gone so we must
    # clear person_id from config to leave the pet in a consistent (unlinked) state.
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(f"{imm.IMMICH_URL}/api/people", headers=imm.headers(), json={"name": name})
    except httpx.ConnectError:
        config[name]["person_id"] = None
        data.save_config(config, DATA_DIR)
        raise HTTPException(status_code=503, detail="Cannot reach Immich. Is it running? Pet has been unlinked.")
    except httpx.TimeoutException:
        config[name]["person_id"] = None
        data.save_config(config, DATA_DIR)
        raise HTTPException(status_code=504, detail="Immich did not respond in time. Pet has been unlinked.")
    if resp.status_code not in (200, 201):
        config[name]["person_id"] = None
        data.save_config(config, DATA_DIR)
        raise HTTPException(
            status_code=resp.status_code,
            detail=f"Immich error creating person: {resp.text}. Pet '{name}' has been unlinked from Immich. Re-create it from the pet settings.",
        )

    new_person_id = resp.json().get("id")
    if not new_person_id:
        config[name]["person_id"] = None
        data.save_config(config, DATA_DIR)
        raise HTTPException(status_code=502, detail="Immich returned no person ID. Pet has been unlinked. Re-create it from the pet settings.")

    # Clean up the old pet folder now that refs were already loaded above.
    if old_person_id:
        old_dir = PETS_DIR / old_person_id
        if old_dir.exists():
            try:
                shutil.rmtree(old_dir)
            except Exception as e:
                log.warning(f"Could not remove old pet folder {old_dir}: {e}")

    (PETS_DIR / new_person_id).mkdir(parents=True, exist_ok=True)
    cleaned_refs = [
        {"asset_id": r["asset_id"], "crop_idx": r.get("crop_idx"), "bbox": r.get("bbox"), "face_id": None}
        for r in old_refs
    ]
    data.save_pet_refs(new_person_id, cleaned_refs, DATA_DIR)

    config[name]["person_id"] = new_person_id
    data.save_config(config, DATA_DIR)

    log.info(f"Reset pet '{name}': old_person={old_person_id}, new_person={new_person_id}, refs_preserved={len(old_refs)}")
    return {"ok": True, "new_person_id": new_person_id, "refs_preserved": len(old_refs)}


@router.delete("/pets/{name}")
async def delete_pet(name: str, local_only: bool = False):
    config = data.load_config(DATA_DIR)
    if name not in config:
        raise HTTPException(status_code=404, detail=f"Pet '{name}' not found")
    person_id = config[name].get("person_id")

    if not local_only and person_id:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.delete(f"{imm.IMMICH_URL}/api/people/{person_id}", headers=imm.headers())
            if resp.status_code not in (200, 204):
                raise HTTPException(status_code=resp.status_code, detail=f"Immich error: {resp.text}")
            log.info(f"Deleted Immich person {person_id} for pet '{name}', face cleanup running in background")
            await _untag_pet_refs(client, config, name, data.load_pet_refs(person_id, DATA_DIR))

    del config[name]
    data.save_config(config, DATA_DIR)
    pet_dir = PETS_DIR / (person_id or name)
    if pet_dir.exists():
        shutil.rmtree(pet_dir)
    log.info(f"Deleted pet '{name}' (local_only={local_only})")
    return {"ok": True}


# ---------------------------------------------------------------------------
# Negatives
# ---------------------------------------------------------------------------

def _crop_thumb(asset_id: str, bbox) -> str:
    return f"/api/crop/{asset_id}?bbox={','.join(str(v) for v in bbox)}" if bbox else f"/api/thumb/{asset_id}"


@router.get("/negatives")
async def get_negatives():
    entries = data.load_negatives(DATA_DIR)
    return {
        "assets": [{"id": e["asset_id"], "bbox": e["bbox"], "thumb": _crop_thumb(e["asset_id"], e["bbox"])} for e in entries],
        "count": len(entries),
    }


def _append_negatives(entries: list) -> tuple[list[dict], int]:
    """Normalize entries, merge them into negatives.json and save. Returns (merged, added)."""
    existing = data.load_negatives(DATA_DIR)
    merged = data.merge_crop_entries(existing, data.normalize_crop_entries(entries))
    data.save_negatives(merged, DATA_DIR)
    return merged, len(merged) - len(existing)


@router.post("/negatives")
async def add_negatives(body: NegativesBody):
    merged, added = _append_negatives(
        [{"asset_id": e.asset_id, "bbox": e.bbox} for e in body.entries] + list(body.asset_ids)
    )
    log.info(f"Negatives: {len(merged)} total (+{added} new)")
    return {"ok": True, "count": len(merged)}


@router.post("/negatives/batch")
async def add_negatives_batch(body: PetAssets):
    """Validate and add multiple asset IDs as negatives, loading YOLO/CLIP once
    for the whole batch instead of once per asset like the single-add path."""
    ids = list(dict.fromkeys(body.asset_ids))  # de-dupe, preserve order

    async def asset_exists(asset_id: str) -> bool:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(f"{imm.IMMICH_URL}/api/assets/{asset_id}", headers=imm.headers())
        return resp.status_code == 200

    exists_flags = await asyncio.gather(*(asset_exists(aid) for aid in ids))
    valid_ids = [aid for aid, ok in zip(ids, exists_flags) if ok]
    invalid_ids = [aid for aid, ok in zip(ids, exists_flags) if not ok]

    def warm_cache():
        with inference_session():
            for aid in valid_ids:
                try:
                    emb.get_crops_and_embed(aid)
                except Exception as e:
                    log.warning(f"Could not embed asset {aid} for negatives batch: {e}")

    if valid_ids:
        await asyncio.to_thread(warm_cache)

    merged, added = _append_negatives(valid_ids)
    log.info(f"Negatives batch: {len(merged)} total (+{added} new, {len(invalid_ids)} invalid)")
    return {"ok": True, "count": len(merged), "added": added, "invalid_ids": invalid_ids}


@router.delete("/pets/{name}/refs")
async def clear_pet_refs(name: str):
    config = data.load_config(DATA_DIR)
    if name not in config:
        raise HTTPException(status_code=404, detail=f"Pet '{name}' not found")
    person_id = config[name].get("person_id") or name
    data.save_pet_refs(person_id, [], DATA_DIR)
    log.info(f"Cleared all refs for pet '{name}' (local only)")
    return {"ok": True}


@router.delete("/negatives/all")
async def clear_all_negatives():
    data.save_negatives([], DATA_DIR)
    log.info("Cleared all negatives (local only)")
    return {"ok": True}


@router.delete("/negatives/{asset_id}")
async def remove_negative(asset_id: str, bbox: Optional[str] = None):
    """Without bbox, removes every entry of the asset. With bbox=x1,y1,x2,y2, removes only
    that crop entry. bbox=whole removes only the whole-photo entry."""
    entries = data.load_negatives(DATA_DIR)
    if bbox is None:
        keep = [e for e in entries if e["asset_id"] != asset_id]
    else:
        target_bbox = None
        if bbox != "whole":
            try:
                target_bbox = [float(v) for v in bbox.split(",")]
            except ValueError:
                target_bbox = []
            if len(target_bbox) != 4:
                raise HTTPException(status_code=400, detail="bbox must be x1,y1,x2,y2 or whole")
        target = data.crop_entry_key({"asset_id": asset_id, "bbox": target_bbox})
        keep = [e for e in entries if data.crop_entry_key(e) != target]
    data.save_negatives(keep, DATA_DIR)
    return {"ok": True}


# ---------------------------------------------------------------------------
# Skipped
# ---------------------------------------------------------------------------

@router.get("/skipped")
async def get_skipped():
    ids = data.load_skipped_ids(DATA_DIR)
    return {"assets": [{"id": aid, "thumb": _crop_thumb(aid, None)} for aid in ids], "count": len(ids)}


@router.post("/skipped")
async def add_skipped(body: PetAssets):
    existing = data.load_skipped_ids(DATA_DIR)
    merged = list(dict.fromkeys(existing + body.asset_ids))  # keep order, newest last
    data.save_skipped_ids(merged, DATA_DIR)
    return {"count": len(merged)}


@router.delete("/skipped/{asset_id}")
async def remove_skipped(asset_id: str):
    ids = [i for i in data.load_skipped_ids(DATA_DIR) if i != asset_id]
    data.save_skipped_ids(ids, DATA_DIR)
    return {"ok": True}


# ---------------------------------------------------------------------------
# Pet reference assets
# ---------------------------------------------------------------------------

@router.get("/pets/{name}/assets")
async def get_pet_assets(name: str):
    config = data.load_config(DATA_DIR)
    if name not in config:
        raise HTTPException(status_code=404, detail=f"Pet '{name}' not found")
    person_id = config[name].get("person_id") or name
    refs = data.load_pet_refs(person_id, DATA_DIR)

    def make_item(r: dict) -> dict:
        aid = r["asset_id"]
        cidx = r.get("crop_idx")
        bbox = r.get("bbox")
        if bbox:
            thumb = f"/api/crop/{aid}?bbox={','.join(str(v) for v in bbox)}"
        else:
            thumb = f"/api/crop/{aid}"
        return {"id": aid, "crop_idx": cidx, "bbox": bbox, "thumb": thumb}

    return {"assets": [make_item(r) for r in refs]}


@router.post("/pets/{name}/assets")
async def set_pet_assets(name: str, body: PetCropAssets):
    config = data.load_config(DATA_DIR)
    if name not in config:
        raise HTTPException(status_code=404, detail=f"Pet '{name}' not found")
    person_id = config[name].get("person_id")
    folder_key = person_id or name

    # Normalize to list of CropRef
    if body.assets is not None:
        crop_refs = body.assets
    elif body.asset_ids is not None:
        crop_refs = [CropRef(asset_id=aid) for aid in body.asset_ids]
    else:
        crop_refs = []

    # Build lookup: asset_id -> existing ref (for face_id retrieval)
    existing_refs_by_id: dict[str, dict] = {}
    for r in data.load_pet_refs(folder_key, DATA_DIR):
        existing_refs_by_id.setdefault(r["asset_id"], r)
    existing_asset_ids = set(existing_refs_by_id.keys())

    # Determine new asset_ids (need face assignment, deduplicated)
    seen_aids: set[str] = set()
    new_asset_ids: list[str] = []
    bbox_by_aid: dict[str, list[float]] = {}
    for cr in crop_refs:
        if cr.bbox and cr.asset_id not in bbox_by_aid:
            bbox_by_aid[cr.asset_id] = cr.bbox
        if cr.asset_id not in existing_asset_ids and cr.asset_id not in seen_aids:
            seen_aids.add(cr.asset_id)
            new_asset_ids.append(cr.asset_id)

    log.info(f"Saving {len(crop_refs)} refs for pet '{name}' ({len(new_asset_ids)} new assets)")

    ok = fail = skipped = 0
    new_face_ids: dict[str, str] = {}

    if person_id and new_asset_ids:
        async with httpx.AsyncClient(timeout=30) as client:
            for aid in new_asset_ids:
                existing_persons = await imm.get_existing_face_person_ids(client, aid)
                if person_id in existing_persons:
                    skipped += 1
                    continue
                face_id = await imm.post_face(client, aid, person_id, bbox_by_aid.get(aid))
                if face_id:
                    new_face_ids[aid] = face_id
                    ok += 1
                else:
                    fail += 1
        log.info(f"Face assignment for '{name}': {ok} ok, {fail} failed, {skipped} already present")
    elif not person_id:
        log.warning(f"Pet '{name}' has no person_id, skipping face assignment")

    final_refs = []
    for cr in crop_refs:
        face_id = new_face_ids.get(cr.asset_id) or existing_refs_by_id.get(cr.asset_id, {}).get("face_id")
        final_refs.append({
            "asset_id": cr.asset_id,
            "crop_idx": cr.crop_idx,
            "bbox": cr.bbox,
            "face_id": face_id,
        })
    data.save_pet_refs(folder_key, final_refs, DATA_DIR)
    return {"ok": True, "count": len(final_refs), "faces_added": ok, "faces_failed": fail}


def _other_pets_have_asset(config: dict, exclude_name: str, asset_id: str) -> bool:
    """True if some pet other than exclude_name still has a ref on asset_id, meaning the
    photo still carries a face this tool wrote and shouldn't lose its review tag."""
    for pet_name, cfg in config.items():
        if pet_name == exclude_name:
            continue
        folder_key = cfg.get("person_id") or pet_name
        if any(r["asset_id"] == asset_id for r in data.load_pet_refs(folder_key, DATA_DIR)):
            return True
    return False


async def _untag_pet_refs(client: httpx.AsyncClient, config: dict, name: str, refs: list[dict]) -> None:
    """Strip the review tag from every ref's asset after that pet's Immich person is gone,
    skipping any asset another pet still has a face on. Shared by delete_pet and
    reset_pet_immich so the untag step can't drift out of sync between the two."""
    for r in refs:
        if not _other_pets_have_asset(config, name, r["asset_id"]):
            await imm.remove_review_tag(client, r["asset_id"])


@router.delete("/pets/{name}/assets/{asset_id}")
async def remove_pet_asset(name: str, asset_id: str, crop_idx: Optional[int] = None):
    config = data.load_config(DATA_DIR)
    if name not in config:
        raise HTTPException(status_code=404, detail=f"Pet '{name}' not found")
    folder_key = config[name].get("person_id") or name

    refs = data.load_pet_refs(folder_key, DATA_DIR)

    if crop_idx is not None:
        remaining = [r for r in refs if not (r["asset_id"] == asset_id and r.get("crop_idx") == crop_idx)]
        still_has_asset = any(r["asset_id"] == asset_id for r in remaining)
    else:
        remaining = [r for r in refs if r["asset_id"] != asset_id]
        still_has_asset = False

    if not still_has_asset:
        removed = [r for r in refs if r["asset_id"] == asset_id]
        face_id = removed[0].get("face_id") if removed else None
        if face_id:
            async with httpx.AsyncClient(timeout=15) as client:
                status = await imm.delete_face(client, face_id, asset_id, untag=not _other_pets_have_asset(config, name, asset_id))
            log.info(f"Deleted face {face_id} on asset {asset_id} for pet '{name}' (status={status})")
        else:
            log.warning(f"No stored face_id for asset {asset_id} on pet '{name}', face not removed from Immich")

    data.save_pet_refs(folder_key, remaining, DATA_DIR)
    return {"ok": True}



# ---------------------------------------------------------------------------
# Ref suggestions
# ---------------------------------------------------------------------------

def _build_classifier_from_config(config: dict):
    """Load all pet refs and return (names, clf, scaler), or None if no pets have refs.

    The trained classifier is cached in memory and only rebuilt when the set of
    refs or negatives changes. This avoids re-embedding every ref on each request
    and keeps prediction scores stable between calls."""
    from classifier import build_classifier, classifier_fingerprint
    all_pet_names = list(config.keys())
    all_refs = {n: data.load_pet_refs(config[n].get("person_id") or n, DATA_DIR) for n in all_pet_names}
    pet_names = [n for n in all_pet_names if all_refs.get(n)]
    refs_per_pet = {n: all_refs[n] for n in pet_names}
    negatives = data.load_negatives(DATA_DIR)

    fp = classifier_fingerprint(pet_names, refs_per_pet, negatives)
    with state.classifier_cache_lock:
        cached = state.classifier_cache
        if cached is not None and cached["fingerprint"] == fp:
            return cached["names"], cached["clf"], cached["scaler"]

    result = build_classifier(pet_names, refs_per_pet, negatives)
    if result is None:
        return None
    names, clf, scaler = result
    with state.classifier_cache_lock:
        state.classifier_cache = {"fingerprint": fp, "names": names, "clf": clf, "scaler": scaler}
    return names, clf, scaler


def _finder_error(summary: str, errors_before: int) -> RuntimeError:
    """Error for a finder that could not score anything: summary plus the model failure
    recorded since errors_before, if any. build_response turns it into a 503."""
    reason = describe_model_failure(errors_before)
    return RuntimeError(" ".join(p for p in (summary, reason) if p))


@router.get("/pets/{name}/suggestions")
async def get_suggestions(name: str, limit: int = 20):
    config = data.load_config(DATA_DIR)
    if name not in config:
        raise HTTPException(status_code=404, detail=f"Pet '{name}' not found")

    pet_cfg = config[name]
    description = pet_cfg.get("description", "").strip()
    if not description:
        raise HTTPException(status_code=400, detail="no_description")

    ref_ids = data.load_pet_asset_ids(pet_cfg.get("person_id") or name, DATA_DIR)
    ref_set = set(ref_ids)
    # Whole-photo negatives exclude the asset up front, crop negatives only their crop below.
    neg_index = data.index_crop_entries(data.load_negatives(DATA_DIR))
    exclude = ref_set | data.whole_photo_ids(neg_index) | set(data.load_skipped_ids(DATA_DIR))

    async with httpx.AsyncClient(timeout=30) as client:
        if ref_ids:
            candidates = await _visual_search(client, ref_ids, pet_cfg, exclude)
        else:
            # Ask for enough to still have results once the excluded photos are filtered out.
            body: dict = {"query": description, "size": min(1000, 60 + len(exclude)), **_pet_image_search_params(pet_cfg)}
            resp = await client.post(f"{imm.IMMICH_URL}/api/search/smart", headers=imm.headers(), json=body)
            if resp.status_code != 200:
                raise HTTPException(status_code=resp.status_code, detail=resp.text)
            all_items = resp.json().get("assets", {}).get("items", [])
            candidates = [a for a in all_items if a["id"] not in exclude]
            candidates = await asyncio.to_thread(imm.drop_foreign_assets, candidates, "text search")

    if not candidates:
        return {"assets": []}

    if not ref_ids:
        return {"assets": [_slim_asset(a) for a in candidates[:limit]]}

    def compute():
        with inference_session():
            result = _build_classifier_from_config(config)
            if result is None:
                return []
            names, clf, scaler = result
            if name not in names:
                return []
            pet_idx = names.index(name)
            scored = []
            with ThreadPoolExecutor(max_workers=emb.SCAN_WORKERS) as ex:
                futures = {ex.submit(emb.get_crops_and_embed, a["id"]): a for a in candidates}
                for future in as_completed(futures):
                    a = futures[future]
                    for c, vec in (future.result() or []):
                        if data.crop_entry_matches(neg_index, a["id"], c["bbox"]):
                            continue
                        v = np.asarray(vec, dtype=np.float64).reshape(1, -1)
                        prob = float(clf.predict_proba(scaler.transform(v))[0][pet_idx])
                        scored.append((prob, {**_slim_asset(a), "crops": [c]}))
            scored.sort(key=lambda x: x[0], reverse=True)
            return [item for _, item in scored[:limit]]

    async def build_response():
        try:
            results = await asyncio.wait_for(asyncio.to_thread(compute), timeout=LONG_REQUEST_TIMEOUT)
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail=f"Timed out after {LONG_REQUEST_TIMEOUT}s")
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))
        return {"assets": results}

    return await _streaming_json(build_response())


@router.get("/pets/{name}/borderline")
async def get_borderline(name: str, limit: int = 40):
    from poller import THRESHOLD
    config = data.load_config(DATA_DIR)
    if name not in config:
        raise HTTPException(status_code=404, detail=f"Pet '{name}' not found")

    pet_cfg = config[name]
    ref_ids = data.load_pet_asset_ids(pet_cfg.get("person_id") or name, DATA_DIR)
    if not ref_ids:
        raise HTTPException(status_code=400, detail="no_refs")

    ref_set = set(ref_ids)
    # Whole-photo negatives exclude the asset up front, crop negatives only their crop below.
    neg_index = data.index_crop_entries(data.load_negatives(DATA_DIR))
    skipped_ids = set(data.load_skipped_ids(DATA_DIR))
    exclude = ref_set | data.whole_photo_ids(neg_index) | skipped_ids

    async with httpx.AsyncClient(timeout=30) as client:
        candidates = await _visual_search(client, ref_ids, pet_cfg, exclude)

    if not candidates:
        return {"assets": []}

    LOW, HIGH = 0.3, THRESHOLD

    state.borderline_request_id += 1
    my_id = state.borderline_request_id

    def compute():
        state.borderline_progress["current"] = 0
        state.borderline_progress["total"] = 0
        state.borderline_progress["running"] = True
        try:
            with inference_session():
                errors_before = model_error_count()
                result = _build_classifier_from_config(config)
                if result is None or name not in result[0]:
                    # This pet has refs, so no classifier means they could not be embedded.
                    log.warning(f"borderline for '{name}': classifier unavailable, no reference photo could be embedded")
                    raise _finder_error(f"Could not process any reference photo for '{name}'.", errors_before)
                names, clf, scaler = result
                pet_idx = names.index(name)
                state.borderline_progress["total"] = len(candidates)
                scored = []
                with ThreadPoolExecutor(max_workers=emb.SCAN_WORKERS) as ex:
                    futures = {ex.submit(emb.get_crops_and_embed, a["id"]): a for a in candidates}
                    done = 0
                    for future in as_completed(futures):
                        if state.borderline_request_id != my_id:
                            ex.shutdown(wait=False, cancel_futures=True)
                            return []
                        done += 1
                        state.borderline_progress["current"] = done
                        a = futures[future]
                        for c, vec in (future.result() or []):
                            if data.crop_entry_matches(neg_index, a["id"], c["bbox"]):
                                continue
                            v = np.asarray(vec, dtype=np.float64).reshape(1, -1)
                            pet_prob = float(clf.predict_proba(scaler.transform(v))[0][pet_idx])
                            if LOW <= pet_prob < HIGH:
                                scored.append((pet_prob, {**_slim_asset(a), "crops": [c], "score": round(pet_prob, 3)}))
                if not scored and model_error_count() > errors_before:
                    raise _finder_error("", errors_before)
                scored.sort(key=lambda x: x[0])
                return scored[:limit]
        finally:
            if state.borderline_request_id == my_id:
                state.borderline_progress["running"] = False

    async def build_response():
        try:
            scored = await asyncio.wait_for(asyncio.to_thread(compute), timeout=LONG_REQUEST_TIMEOUT)
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail=f"Timed out after {LONG_REQUEST_TIMEOUT}s")
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))
        return {
            "assets": [slim for _, slim in scored],
            "threshold": THRESHOLD,
        }

    return await _streaming_json(build_response())


@router.get("/pets/{name}/borderline/progress")
async def get_borderline_progress(name: str):
    return state.borderline_progress


BENCHMARK_BUCKETS = ("yolo", "fallback", "video_yolo", "video_fallback")


class BenchmarkRequest(BaseModel):
    since: str
    until: Optional[str] = None
    yolo_conf: Optional[float] = None  # detection confidence floor for this run only


@router.post("/analysis/benchmark")
async def start_benchmark(body: BenchmarkRequest):
    """Diagnostic, not part of the tagging pipeline: dry-run classify every asset (photo
    and video) in the given date range and compare against the actual Immich tags (source
    of truth). Runs in the background since a full library can take several minutes; poll
    GET /analysis/benchmark for progress and the result. See accuracy.html.

    Immich's tags in the range are treated as ground truth, so the caller is responsible
    for making sure every asset in range has already been manually tagged and corrected
    before running this; un-reviewed or wrong tags produce misleading results.

    No CLIP threshold is taken here: unlike yolo_conf (which changes what gets embedded,
    so it has to be fixed before running), a CLIP threshold is just a cutoff compared
    against scores already collected, so it's explored entirely client-side afterward
    (see result['curve'] and accuracy.html's threshold explorer) with no re-run needed."""
    import re
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", body.since):
        raise HTTPException(status_code=400, detail="since must be YYYY-MM-DD")
    if body.yolo_conf is not None and not (0 < body.yolo_conf <= 1):
        raise HTTPException(status_code=400, detail="yolo_conf must be between 0 and 1")
    if state.benchmark_progress["running"]:
        raise HTTPException(status_code=409, detail="A benchmark is already running")
    state.benchmark_generation += 1
    state.benchmark_cancel.clear()
    asyncio.create_task(_run_benchmark(
        state.benchmark_generation, body.since, body.until, body.yolo_conf,
    ))
    return {"status": "started"}


@router.post("/analysis/benchmark/stop")
async def stop_benchmark():
    """Cancels an in-progress benchmark. Whatever was already classified before the
    cancel is still assembled into a usable result (same as a completed run), not
    discarded, since a partial answer over most of the range beats none at all."""
    if state.benchmark_progress["running"]:
        state.benchmark_cancel.set()
    return {"status": "stopping"}


@router.get("/analysis/benchmark")
async def get_benchmark():
    return {"progress": state.benchmark_progress, "result": state.benchmark_result}


@router.post("/analysis/benchmark/result")
async def set_benchmark_result(body: dict):
    """Store an externally-provided result (e.g. a file re-imported in accuracy.html) as
    the current one, so /download can serve it back too, not just freshly-run results."""
    state.benchmark_result = body
    return {"status": "ok"}


@router.get("/analysis/benchmark/download")
async def download_benchmark():
    """Serves the current result as a file attachment via Content-Disposition instead of
    a client-side blob URL: some browsers don't reliably honor the <a download> attribute
    on blob: URLs, silently saving with the blob's internal id and no extension instead.
    Filename encodes the run's YOLO confidence and date range, not just today's date, so
    several downloaded results sitting in the same folder stay distinguishable at a glance
    without having to open each one."""
    if state.benchmark_result is None:
        raise HTTPException(status_code=404, detail="No benchmark result available")
    cfg = state.benchmark_result.get("config") or {}
    yolo_conf = cfg.get("yolo_conf")
    since = cfg.get("since") or "unknown"
    until = cfg.get("until") or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    filename = f"accuracy_yolo{yolo_conf}_{since}_{until}.json"
    return Response(
        content=json.dumps(state.benchmark_result),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


SWEEP_SCORE_FLOOR = 0.3
"""Lower bound for scores kept in the threshold-sweep curve data (result['curve']). Ground
truth items are always kept regardless of score, this only bounds how far below the chosen
threshold a non-ground-truth score is worth recording; below it the classifier is unusable
anyway (existing convention: get_neg_candidates also treats 0.30 as the floor for
'plausible enough to matter'), and keeping every near-zero score for every asset would bloat
the result far beyond what an interactive sweep from a sane range needs. This is the only
threshold-shaped constant the backend applies; the actual CLIP tagging threshold is picked
entirely client-side against this data, see start_benchmark's docstring."""


def _benchmark_bucket(asset_type: str, yolo_detected: bool) -> str:
    """Split by detected vs whole-image fallback, and by photo vs video (a video only ever
    classifies a single sampled frame, so its numbers behave differently from a photo's).
    A strong/weak split by detection confidence was tried and measured worse: at matched
    recall, splitting out a separate 'weak' tier and giving it its own threshold produced
    a higher false-positive rate than just raising the YOLO_CONF floor and using one
    threshold for every detection."""
    is_video = asset_type == "VIDEO"
    if not yolo_detected:
        return "video_fallback" if is_video else "fallback"
    return "video_yolo" if is_video else "yolo"


async def _run_benchmark(
    generation: int, since: str, until: str | None,
    yolo_conf_override: float | None = None,
):
    state.benchmark_progress = {"current": 0, "total": 0, "running": True}
    # Deliberately not clearing state.benchmark_result here: whatever result was already
    # there (a prior run, or one imported via /analysis/benchmark/result) stays visible
    # while this run is in progress, and is only replaced below once this run actually
    # produced something, so stopping a run almost immediately can't wipe out a perfectly
    # good previous result and leave the page with nothing.

    def compute():
        yolo_conf = yolo_conf_override if yolo_conf_override is not None else det.YOLO_CONF
        config_block = {
            "clip_model": emb.CLIP_MODEL_NAME, "clip_pretrained": emb.CLIP_PRETRAINED,
            "yolo_model": det.YOLO_MODEL_NAME, "yolo_input_size": det.YOLO_INPUT_SIZE,
            "yolo_conf": yolo_conf, "since": since, "until": until,
        }
        with inference_session():
            config = data.load_config(DATA_DIR)
            pet_names = [n for n, cfg in config.items() if cfg.get("person_id")]
            person_to_pet = {config[n]["person_id"]: n for n in pet_names}
            result = _build_classifier_from_config(config)
            if result is None:
                return {"config": config_block, "curve": {}}
            names, clf, scaler = result
            pet_idx = {n: names.index(n) for n in pet_names if n in names}

            since_iso = since + "T00:00:00.000Z"
            until_iso = f"{until}T23:59:59.999Z" if until else None
            assets = imm.fetch_assets_in_range(since_iso, until_iso)
            state.benchmark_progress["total"] = len(assets)

            def process(a):
                """Returns (bucket, pet, gt, item) for every pet that either has ground
                truth on this asset or scored high enough to matter for the threshold
                explorer (see SWEEP_SCORE_FLOOR). No threshold is applied here at all,
                that's picked client-side against these scores. Pure per-asset work only,
                no shared state, so results are safe to aggregate after the fact in the
                single-threaded collection loop below."""
                aid = a.get("id")
                gt_pets = {person_to_pet[p] for p in imm.fetch_asset_face_person_ids(aid) if p in person_to_pet}
                img, detected = emb.crop_animals_cached(aid, conf=yolo_conf, with_conf=True)
                if img is None:
                    return []
                yolo_detected = bool(detected)
                max_det_conf = max((c for _, _, c in detected), default=None)
                crops = [(bbox, crop) for bbox, crop, _ in detected] if detected else [(None, img)]
                best = {n: 0.0 for n in pet_names}
                for _, crop in crops:
                    vec = emb.embed_image(crop)
                    if vec is None:
                        continue
                    v = np.asarray(vec, dtype=np.float64).reshape(1, -1)
                    probs = clf.predict_proba(scaler.transform(v))[0]
                    for n in pet_names:
                        if n in pet_idx:
                            best[n] = max(best[n], float(probs[pet_idx[n]]))
                bucket = _benchmark_bucket(a.get("type"), yolo_detected)
                rows = []
                for n in pet_names:
                    gt = n in gt_pets
                    score = best[n]
                    if not (gt or score >= SWEEP_SCORE_FLOOR):
                        continue
                    item = {"asset_id": aid, "date": (a.get("fileCreatedAt") or "")[:10],
                             "type": a.get("type"), "yolo_detected": yolo_detected, "bucket": bucket,
                             "det_conf": round(max_det_conf, 4) if max_det_conf is not None else None,
                             "score": round(score, 4)}
                    rows.append((bucket, n, gt, item))
                return rows

            curve: dict[tuple[str, str], dict[str, list]] = {}

            cancelled = False
            with ThreadPoolExecutor(max_workers=emb.SCAN_WORKERS) as ex:
                futures = [ex.submit(process, a) for a in assets]
                for i, fut in enumerate(as_completed(futures), 1):
                    for bucket, pet, gt, item in fut.result():
                        cc = curve.setdefault((bucket, pet), {"gt": [], "extra": []})
                        cc["gt" if gt else "extra"].append(item)
                    state.benchmark_progress["current"] = i
                    if state.benchmark_cancel.is_set():
                        cancelled = True
                        ex.shutdown(wait=False, cancel_futures=True)
                        break

            config_block["partial"] = cancelled
            if cancelled:
                config_block["assets_scanned"] = state.benchmark_progress["current"]
                config_block["assets_total"] = len(assets)
            curve_out = {bucket: {} for bucket in BENCHMARK_BUCKETS}
            for (bucket, pet), items in curve.items():
                for lst in items.values():
                    lst.sort(key=lambda x: -x["score"])
                curve_out.setdefault(bucket, {})[pet] = items
            return {"config": config_block, "curve": curve_out}

    try:
        new_result = await asyncio.to_thread(compute)
        has_data = any(new_result["curve"].get(b) for b in BENCHMARK_BUCKETS)
        # Only replace the previous result if this run collected something, or actually
        # ran to completion uncancelled (an empty range is a legitimate result). A run
        # stopped before it collected any data leaves the previous result in place instead
        # of overwriting it with an empty one, see the comment above.
        if has_data or not new_result["config"].get("partial"):
            state.benchmark_result = new_result
    finally:
        state.benchmark_progress["running"] = False


# ---------------------------------------------------------------------------
# Reference photo analysis
#
# Diagnostic (not part of the tagging pipeline) that scores each reference photo by
# how much it contributes to the classifier, so the user can prune dead-weight refs
# and spot bad crops. Two independent signals per ref, see refanalysis.html:
#   - redundancy: max cosine similarity of this ref's CLIP embedding to the pet's
#     OTHER refs. High means it is a near-duplicate carrying little new information.
#   - leave-one-out (LOO): drop this one ref, retrain, and measure two things:
#       * self_score  - the model's predicted probability for the pet on this ref's
#         own crop when it was NOT trained on it. Low means an outlier the rest of
#         the refs don't explain: a genuinely informative hard case, or a bad crop.
#       * influence   - how much the other refs' correct-class probability drops when
#         this ref is removed. Positive = load-bearing (removing it hurts the pet);
#         near zero = contributes nothing; negative = removing it actually helps, i.e.
#         it is pulling the boundary the wrong way (a likely bad/mislabeled ref).
# Ref sets are tiny (dozens per pet) so full LOO retraining is cheap; it still runs in
# the background with progress since embedding uncached refs needs the CLIP workers.
# ---------------------------------------------------------------------------

@router.post("/analysis/refs")
async def start_ref_analysis():
    """Kick off the reference-photo analysis over the currently enrolled pets and
    negatives (no parameters: it always reflects the live refs). Runs in the background;
    poll GET /analysis/refs for progress and the result. See refanalysis.html."""
    if state.refanalysis_progress["running"]:
        raise HTTPException(status_code=409, detail="A ref analysis is already running")
    state.refanalysis_generation += 1
    state.refanalysis_cancel.clear()
    asyncio.create_task(_run_ref_analysis(state.refanalysis_generation))
    return {"status": "started"}


@router.post("/analysis/refs/stop")
async def stop_ref_analysis():
    """Cancel an in-progress ref analysis. Refs already scored keep their numbers; the
    rest are returned with null LOO scores and the result is flagged partial."""
    if state.refanalysis_progress["running"]:
        state.refanalysis_cancel.set()
    return {"status": "stopping"}


@router.get("/analysis/refs")
async def get_ref_analysis():
    return {"progress": state.refanalysis_progress, "result": state.refanalysis_result}


@router.post("/analysis/negatives")
async def start_neg_analysis():
    """Kick off the "not a pet" analysis over the current negatives (no parameters). Runs in
    the background; poll GET /analysis/negatives for progress and the result. Separate from the
    ref analysis: see _run_neg_analysis. See refanalysis.html."""
    if state.neganalysis_progress["running"]:
        raise HTTPException(status_code=409, detail="A negatives analysis is already running")
    state.neganalysis_generation += 1
    state.neganalysis_cancel.clear()
    asyncio.create_task(_run_neg_analysis(state.neganalysis_generation))
    return {"status": "started"}


@router.post("/analysis/negatives/stop")
async def stop_neg_analysis():
    """Cancel an in-progress negatives analysis. Negatives already scored keep their numbers;
    the rest come back with null scores and the result is flagged partial."""
    if state.neganalysis_progress["running"]:
        state.neganalysis_cancel.set()
    return {"status": "stopping"}


@router.get("/analysis/negatives")
async def get_neg_analysis():
    return {"progress": state.neganalysis_progress, "result": state.neganalysis_result}


def _normalize_rows(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


# --- Stateless helpers shared by the ref analysis and the negatives analysis. They take an
# explicit (X, y) so each analysis can build its own model without reaching into the other. ---

def _ra_config_block() -> dict:
    return {
        "clip_model": emb.CLIP_MODEL_NAME, "clip_pretrained": emb.CLIP_PRETRAINED,
        "yolo_model": det.YOLO_MODEL_NAME, "yolo_input_size": det.YOLO_INPUT_SIZE,
    }


def _ra_embed_pet_refs(data_dir):
    """Embed every enrolled pet's reference photos. Returns (pet_names, ref_records,
    pet_vecs, pet_row_labels); pet_vecs are the CLIP vectors in ref order, so pet rows
    occupy indices 0..len(pet_vecs)-1 in any model built on top of them, and each ref_record
    carries the row indices it contributed so a single ref can be held out."""
    config = data.load_config(data_dir)
    all_pet_names = list(config.keys())
    refs_per_pet = {n: data.load_pet_refs(config[n].get("person_id") or n, data_dir)
                    for n in all_pet_names}
    pet_names = [n for n in all_pet_names if refs_per_pet.get(n)]
    pet_vecs: list = []
    pet_row_labels: list = []
    ref_records: list[dict] = []
    for i, name in enumerate(pet_names):
        for ref in refs_per_pet[name]:
            asset_id = ref["asset_id"]
            bbox = ref.get("bbox")
            if bbox:
                vec = emb.embed_crop_by_bbox(asset_id, bbox)
                vecs = [vec] if vec is not None else []
            else:
                vecs = emb.embed_asset_crops(asset_id, require_animal=True)
                if not vecs:
                    vecs = emb.embed_asset_crops(asset_id, require_animal=False)
            rows = list(range(len(pet_vecs), len(pet_vecs) + len(vecs)))
            pet_vecs.extend(vecs)
            pet_row_labels.extend([i] * len(vecs))
            ref_records.append({"pet": name, "asset_id": asset_id,
                                "crop_idx": ref.get("crop_idx"), "bbox": bbox,
                                "rows": rows, "vecs": list(vecs)})
    return pet_names, ref_records, pet_vecs, pet_row_labels


def _ra_fit(X, y, mask):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    from classifier import fit_logreg
    scaler = StandardScaler()
    clf = LogisticRegression(max_iter=1000, random_state=0)
    fit_logreg(clf, scaler.fit_transform(X[mask]), y[mask])
    return clf, scaler


def _ra_correct_probs(X, y, clf, scaler, rows_idx):
    """Predicted probability of each row's own (correct) class."""
    if len(rows_idx) == 0:
        return np.array([])
    P = clf.predict_proba(scaler.transform(X[rows_idx]))
    col = {int(c): k for k, c in enumerate(clf.classes_)}
    out = np.zeros(len(rows_idx))
    for j, ri in enumerate(rows_idx):
        c = col.get(int(y[ri]))
        if c is not None:
            out[j] = P[j, c]
    return out


def _ra_correct_margins(X, y, clf, scaler, rows_idx):
    """Log-odds (decision_function) of each row's own class. Unlike probability, this does not
    saturate near 1.0 on well-separated refs, so removing a ref still produces a measurable
    shift; influence is measured in this space."""
    if len(rows_idx) == 0:
        return np.array([])
    D = clf.decision_function(scaler.transform(X[rows_idx]))
    if D.ndim == 1:  # binary case: score is for classes_[1]; classes_[0] is its negation
        D = np.column_stack([-D, D])
    col = {int(c): k for k, c in enumerate(clf.classes_)}
    out = np.zeros(len(rows_idx))
    for j, ri in enumerate(rows_idx):
        c = col.get(int(y[ri]))
        if c is not None:
            out[j] = D[j, c]
    return out


def _ra_pet_probs(X, clf, scaler, rows_idx, pet_labels):
    """For each row, the predicted probability of every pet class (columns aligned to
    pet_labels). Used to ask how pet-like a negative looks when it is held out."""
    P = clf.predict_proba(scaler.transform(X[rows_idx]))
    col = {int(c): k for k, c in enumerate(clf.classes_)}
    out = np.zeros((len(rows_idx), len(pet_labels)))
    for m, lbl in enumerate(pet_labels):
        c = col.get(int(lbl))
        if c is not None:
            out[:, m] = P[:, c]
    return out


def _ra_set_redundancy(group):
    """Fill redundancy (max cosine to another member) + nearest_asset_id on records that carry
    a normalized 'mean' embedding. group is a list of record dicts."""
    for r in group:
        r.setdefault("redundancy", None)
        r.setdefault("nearest_asset_id", None)
    if len(group) < 2:
        return
    M = np.array([r["mean"] for r in group])
    sims = M @ M.T
    np.fill_diagonal(sims, -1.0)
    for k, r in enumerate(group):
        j = int(np.argmax(sims[k]))
        r["redundancy"] = round(float(sims[k, j]), 4)
        r["nearest_asset_id"] = group[j]["asset_id"]


async def _run_ref_analysis(generation: int):
    state.refanalysis_progress = {"current": 0, "total": 0, "running": True}

    def compute():
        from classifier import embed_negative, subsample_negatives
        config_block = _ra_config_block()
        empty = {"config": {**config_block, "n_pets": 0, "n_refs": 0,
                            "n_negatives": 0, "n_ref_vecs": 0, "partial": False}, "pets": {}}

        with inference_session():
            pet_names, ref_records, pet_vecs, pet_row_labels = _ra_embed_pet_refs(DATA_DIR)
            n_pet_rows = len(pet_vecs)
            if not pet_names or n_pet_rows == 0:
                return empty
            unknown_idx = len(pet_names)

            # Negatives -> unknown class, subsampled identically to build_classifier so the
            # decision boundary (and every score below) matches the production model.
            negatives = subsample_negatives(data.load_negatives(DATA_DIR), len(ref_records))
            X_list = list(pet_vecs)
            y_list = list(pet_row_labels)
            for entry in negatives:
                for vec in embed_negative(entry):
                    X_list.append(vec)
                    y_list.append(unknown_idx)
            n_neg_rows = len(X_list) - n_pet_rows

            X = np.array(X_list, dtype=np.float64)
            y = np.array(y_list, dtype=np.intp)
            if unknown_idx not in y:
                # Guarantee the unknown class exists (same guard as build_classifier).
                X = np.vstack([X, np.zeros((1, X.shape[1]))])
                y = np.append(y, unknown_idx)

            full_clf, full_scaler = _ra_fit(X, y, np.ones(len(y), dtype=bool))
            full_margins = _ra_correct_margins(X, y, full_clf, full_scaler, np.arange(n_pet_rows))

            # Redundancy: within each pet, cosine similarity between refs' mean embeddings.
            per_pet_refs: dict[str, list[dict]] = {n: [] for n in pet_names}
            for rr in ref_records:
                if rr["vecs"]:
                    rr["mean"] = _normalize_rows(np.array(rr["vecs"], dtype=np.float64).mean(axis=0, keepdims=True))[0]
                    per_pet_refs[rr["pet"]].append(rr)
            for group in per_pet_refs.values():
                _ra_set_redundancy(group)

            pet_total_rows: dict[str, int] = {n: 0 for n in pet_names}
            for rr in ref_records:
                pet_total_rows[rr["pet"]] += len(rr["rows"])

            analyzable = [rr for rr in ref_records if rr["rows"]]
            state.refanalysis_progress["total"] = len(analyzable)
            pet_row_set = set(range(n_pet_rows))
            cancelled = False
            done = 0
            for rr in ref_records:
                rr.setdefault("redundancy", None)
                rr.setdefault("nearest_asset_id", None)
                rr["self_score"] = None
                rr["influence"] = None
                if not rr["rows"] or cancelled:
                    continue
                # Skip if removing this ref would erase its pet's class entirely.
                if pet_total_rows[rr["pet"]] > len(rr["rows"]):
                    mask = np.ones(len(y), dtype=bool)
                    mask[rr["rows"]] = False
                    loo_clf, loo_scaler = _ra_fit(X, y, mask)
                    # Each of this ref's rows is labeled its pet class, so correct_probs on them
                    # is the held-out own-class probability.
                    self_p = _ra_correct_probs(X, y, loo_clf, loo_scaler, rr["rows"])
                    rr["self_score"] = round(float(self_p.mean()), 4) if len(self_p) else None
                    eval_rows = sorted(pet_row_set - set(rr["rows"]))
                    if eval_rows:
                        # Influence = how much the pet's OTHER refs lose correct-class log-odds
                        # when this ref is dropped. Positive = load-bearing; negative = removing
                        # it helps. 5 decimals: on well-separated sets even a real influence is
                        # small, so coarser rounding flattens it.
                        base = float(full_margins[eval_rows].mean())
                        loo = float(_ra_correct_margins(X, y, loo_clf, loo_scaler, eval_rows).mean())
                        rr["influence"] = round(base - loo, 5)
                done += 1
                state.refanalysis_progress["current"] = done
                if state.refanalysis_cancel.is_set():
                    cancelled = True

            pets_out: dict[str, dict] = {}
            for name in pet_names:
                pets_out[name] = {"refs": [
                    {"asset_id": rr["asset_id"], "crop_idx": rr["crop_idx"], "bbox": rr["bbox"],
                     "n_vecs": len(rr["vecs"]), "embedded": bool(rr["vecs"]),
                     "redundancy": rr["redundancy"], "nearest_asset_id": rr["nearest_asset_id"],
                     "self_score": rr["self_score"], "influence": rr["influence"]}
                    for rr in ref_records if rr["pet"] == name
                ]}

            return {"config": {**config_block, "n_pets": len(pet_names),
                               "n_refs": len(ref_records), "n_negatives": n_neg_rows,
                               "n_ref_vecs": n_pet_rows, "partial": cancelled}, "pets": pets_out}

    try:
        state.refanalysis_result = await asyncio.to_thread(compute)
    finally:
        state.refanalysis_progress["running"] = False


# Cap on how many negatives get the per-negative retrain analysis, to bound runtime; beyond
# this a deterministic subsample is analyzed and the result is flagged subsampled.
NEG_ANALYSIS_CAP = 300


async def _run_neg_analysis(generation: int):
    """Companion to _run_ref_analysis for the "not a pet" set. Kept as its own function (and
    its own model) rather than folded into the ref analysis: a negative's held-out score means
    the opposite of a pet ref's (looking like a pet is what makes a negative valuable), and it
    needs every negative in the unknown class, not the balanced subsample the pet model uses."""
    state.neganalysis_progress = {"current": 0, "total": 0, "running": True}

    def compute():
        import random

        from classifier import embed_negative
        config_block = _ra_config_block()
        empty = {"config": {**config_block, "n_pets": 0, "n_negatives": 0,
                            "n_negatives_analyzed": 0, "neg_subsampled": False, "partial": False},
                 "negatives": {"items": []}}

        with inference_session():
            # Pets define the "looks like a pet" axis; without them there is nothing to score
            # negatives against.
            pet_names, ref_records, pet_vecs, pet_row_labels = _ra_embed_pet_refs(DATA_DIR)
            n_pet_rows = len(pet_vecs)
            if not pet_names or n_pet_rows == 0:
                return empty
            unknown_idx = len(pet_names)
            pet_labels = list(range(len(pet_names)))

            all_negs = sorted(data.load_negatives(DATA_DIR), key=data.crop_entry_key)
            base_config = {**config_block, "n_pets": len(pet_names),
                           "n_negatives": len(all_negs)}
            if not all_negs:
                return {"config": {**base_config, "n_negatives_analyzed": 0,
                                   "neg_subsampled": False, "partial": False},
                        "negatives": {"items": []}}

            neg_subsampled = len(all_negs) > NEG_ANALYSIS_CAP
            negs = (sorted(random.Random(0).sample(all_negs, NEG_ANALYSIS_CAP), key=data.crop_entry_key)
                    if neg_subsampled else all_negs)

            neg_records: list[dict] = []
            neg_vecs: list = []
            for entry in negs:
                vecs = embed_negative(entry)
                rows = list(range(n_pet_rows + len(neg_vecs), n_pet_rows + len(neg_vecs) + len(vecs)))
                neg_vecs.extend(vecs)
                neg_records.append({"asset_id": entry["asset_id"], "bbox": entry.get("bbox"),
                                    "rows": rows, "vecs": list(vecs)})
            n_neg_rows = len(neg_vecs)

            # Model: pet refs + ALL analyzed negatives as the unknown class, so each negative
            # can be held out and asked how pet-like it really is.
            X = np.array(list(pet_vecs) + list(neg_vecs), dtype=np.float64)
            y = np.array(pet_row_labels + [unknown_idx] * n_neg_rows, dtype=np.intp)

            # Normalized centroid of each pet's ref embeddings, for a model-free cosine
            # similarity between a negative and the pet it's confused with (see pet_similarity).
            pet_arr = np.array(pet_vecs, dtype=np.float64)
            pet_lbl_arr = np.array(pet_row_labels)
            pet_centroids = {name: _normalize_rows(pet_arr[pet_lbl_arr == i].mean(axis=0, keepdims=True))[0]
                             for i, name in enumerate(pet_names) if (pet_lbl_arr == i).any()}

            for nr in neg_records:
                if nr["vecs"]:
                    nr["mean"] = _normalize_rows(np.array(nr["vecs"], dtype=np.float64).mean(axis=0, keepdims=True))[0]
            _ra_set_redundancy([nr for nr in neg_records if nr["vecs"]])

            analyzable = [nr for nr in neg_records if nr["rows"]]
            state.neganalysis_progress["total"] = len(analyzable)
            cancelled = False
            done = 0
            for nr in neg_records:
                nr.setdefault("redundancy", None)
                nr.setdefault("nearest_asset_id", None)
                nr["pet_likeness"] = None
                nr["confused_pet"] = None
                nr["pet_similarity"] = None
                if not nr["rows"] or cancelled:
                    continue
                # Skip if this is the only negative (removing it erases the unknown class).
                if n_neg_rows > len(nr["rows"]):
                    mask = np.ones(len(y), dtype=bool)
                    mask[nr["rows"]] = False
                    loo_clf, loo_scaler = _ra_fit(X, y, mask)
                    # Held out, how strongly does the model take this negative for a pet?
                    pp = _ra_pet_probs(X, loo_clf, loo_scaler, nr["rows"], pet_labels).mean(axis=0)
                    best = int(np.argmax(pp))
                    nr["pet_likeness"] = round(float(pp[best]), 4)
                    nr["confused_pet"] = pet_names[best]
                    # Model-free cosine similarity to that pet's photos, as a plain "how close
                    # does it look" number alongside the classifier's pet_likeness.
                    centroid = pet_centroids.get(nr["confused_pet"])
                    if centroid is not None and "mean" in nr:
                        nr["pet_similarity"] = round(float(nr["mean"] @ centroid), 4)
                done += 1
                state.neganalysis_progress["current"] = done
                if state.neganalysis_cancel.is_set():
                    cancelled = True

            items = [
                {"asset_id": nr["asset_id"], "bbox": nr["bbox"],
                 "n_vecs": len(nr["vecs"]), "embedded": bool(nr["vecs"]),
                 "redundancy": nr["redundancy"], "nearest_asset_id": nr["nearest_asset_id"],
                 "pet_likeness": nr["pet_likeness"], "confused_pet": nr["confused_pet"],
                 "pet_similarity": nr["pet_similarity"]}
                for nr in neg_records
            ]
            return {"config": {**base_config, "n_negatives_analyzed": len(neg_records),
                               "neg_subsampled": neg_subsampled, "partial": cancelled},
                    "negatives": {"items": items}}

    try:
        state.neganalysis_result = await asyncio.to_thread(compute)
    finally:
        state.neganalysis_progress["running"] = False


@router.get("/suggestions/negatives")
async def get_neg_candidates(limit: int = 60):
    from poller import THRESHOLD
    config = data.load_config(DATA_DIR)

    all_pet_names = list(config.keys())
    all_refs = {n: data.load_pet_refs(config[n].get("person_id") or n, DATA_DIR) for n in all_pet_names}
    all_ref_ids: set[str] = {r["asset_id"] for refs in all_refs.values() for r in refs}
    # Candidates are whole photos (no crops), so any negative entry, crop or whole photo,
    # excludes its asset here, as before.
    neg_ids = {e["asset_id"] for e in data.load_negatives(DATA_DIR)}
    skipped_ids = set(data.load_skipped_ids(DATA_DIR))
    exclude = all_ref_ids | neg_ids | skipped_ids

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{imm.IMMICH_URL}/api/search/random",
            headers=imm.headers(),
            json={"size": 50, **imm.search_params(asset_type="IMAGE")},
        )
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)

    candidates = [a for a in resp.json() if isinstance(a, dict) and a.get("id") not in exclude]

    if not candidates:
        return {"assets": [], "threshold": THRESHOLD}

    state.neg_request_id += 1
    my_id = state.neg_request_id

    def compute():
        state.neg_progress["current"] = 0
        state.neg_progress["total"] = 0
        state.neg_progress["running"] = True
        try:
            with inference_session():
                errors_before = model_error_count()
                result = _build_classifier_from_config(config)
                if result is None:
                    if all_ref_ids:
                        log.warning("negative candidates: classifier unavailable, no reference photo could be embedded")
                        raise _finder_error("Could not process any reference photo.", errors_before)
                    return []
                names, clf, scaler = result
                unknown_idx = names.index("unknown") if "unknown" in names else -1
                state.neg_progress["total"] = len(candidates)
                scored = []
                for i, a in enumerate(candidates):
                    if state.neg_request_id != my_id:
                        return []
                    state.neg_progress["current"] = i + 1
                    vec = embed_asset(a["id"])
                    if vec is not None:
                        v = np.asarray(vec, dtype=np.float64).reshape(1, -1)
                        probs = clf.predict_proba(scaler.transform(v))[0]
                        pet_prob = (1.0 - float(probs[unknown_idx])) if unknown_idx >= 0 else 0.0
                        if 0.30 <= pet_prob < THRESHOLD:
                            scored.append((pet_prob, a))
                if not scored and model_error_count() > errors_before:
                    raise _finder_error("", errors_before)
                scored.sort(key=lambda x: x[0], reverse=True)
                return scored[:limit]
        finally:
            if state.neg_request_id == my_id:
                state.neg_progress["running"] = False

    async def build_response():
        try:
            scored = await asyncio.wait_for(asyncio.to_thread(compute), timeout=LONG_REQUEST_TIMEOUT)
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail=f"Timed out after {LONG_REQUEST_TIMEOUT}s")
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))
        return {
            "assets": [{**_slim_asset(a), "score": round(prob, 3)} for prob, a in scored],
            "threshold": THRESHOLD,
        }

    return await _streaming_json(build_response())


@router.get("/suggestions/negatives/progress")
async def get_neg_progress():
    return state.neg_progress


# ---------------------------------------------------------------------------
# Scan timestamp
# ---------------------------------------------------------------------------

@router.get("/poll-status")
async def get_poll_status():
    return data.load_poll_status(DATA_DIR)


class PetImport(BaseModel):
    person_id: str
    name: str
    description: str
    since: Optional[str] = None
    until: Optional[str] = None


@router.post("/pets/import")
async def import_pet(body: PetImport):
    name = body.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Name cannot be empty")
    config = data.load_config(DATA_DIR)
    if name.lower() in {k.lower() for k in config}:
        raise HTTPException(status_code=409, detail=f"Pet '{name}' already exists")

    async with httpx.AsyncClient(timeout=15) as client:
        check = await client.get(f"{imm.IMMICH_URL}/api/people/{body.person_id}", headers=imm.headers())
    if check.status_code != 200:
        raise HTTPException(status_code=404, detail="Person not found in Immich")

    candidates: list[tuple[str, str | None]] = []
    async with httpx.AsyncClient(timeout=60) as client:
        search = await client.post(
            f"{imm.IMMICH_URL}/api/search/metadata",
            headers={**imm.headers(), "Content-Type": "application/json"},
            json={"size": 200, **imm.search_params(person_id=body.person_id)},
        )
        if search.status_code == 200:
            block = search.json().get("assets", {})
            items = block.get("items", []) if isinstance(block, dict) else []
            for a in items:
                aid = a.get("id")
                if not aid:
                    continue
                faces_resp = await client.get(f"{imm.IMMICH_URL}/api/faces", headers=imm.headers(), params={"id": aid})
                if faces_resp.status_code == 200:
                    faces = faces_resp.json()
                    named = {f["person"]["id"]: f["id"] for f in faces if f and (f.get("person") or {}).get("id")}
                    if len(named) == 1:
                        candidates.append((aid, named.get(body.person_id)))

    def resolve_all(pairs):
        result = []
        with inference_session():
            with ThreadPoolExecutor(max_workers=emb.SCAN_WORKERS) as ex:
                futures = {ex.submit(emb.resolve_bbox, aid): (aid, face_id) for aid, face_id in pairs}
                for future in as_completed(futures):
                    aid, face_id = futures[future]
                    bbox = future.result()
                    if bbox:
                        result.append((aid, face_id, bbox))
        result.sort(key=lambda x: x[0])
        return result

    try:
        verified = await asyncio.to_thread(resolve_all, candidates)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    n = min(len(verified), 20)
    assets = [
        {"asset_id": verified[int(i * len(verified) / n)][0], "face_id": verified[int(i * len(verified) / n)][1], "bbox": verified[int(i * len(verified) / n)][2]}
        for i in range(n)
    ] if n else []

    (PETS_DIR / body.person_id).mkdir(parents=True, exist_ok=True)
    data.save_pet_refs(body.person_id, assets, DATA_DIR)
    config[name] = {"person_id": body.person_id, "description": body.description, "since": body.since, "until": body.until}
    data.save_config(config, DATA_DIR)
    log.info(f"Imported pet '{name}' from person_id={body.person_id} with {len(assets)} refs")
    return {"name": name, "person_id": body.person_id, "ref_count": len(assets)}


class ScanRequest(BaseModel):
    scan_since: str
    scan_until: Optional[str] = None


@router.post("/scan")
async def trigger_scan(body: ScanRequest):
    import re
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", body.scan_since):
        raise HTTPException(status_code=400, detail="scan_since must be YYYY-MM-DD")
    import state
    if state.scan_lock is not None and state.scan_lock.locked():
        state.scan_cancel.set()
    state.scan_generation += 1
    asyncio.create_task(_run_manual_scan(state.scan_generation, body.scan_since, body.scan_until))
    return {"status": "started"}


@router.post("/scan/stop")
async def stop_scan():
    import state
    if state.scan_lock is not None and state.scan_lock.locked():
        state.scan_cancel.set()
        state.scan_generation += 1
        state.manual_scan_result = {"status": "stopped", "ran_at": datetime.now(timezone.utc).isoformat()}
    return {"status": "stopped"}


async def _run_manual_scan(generation: int, scan_since: str, scan_until: str | None = None):
    import state
    import inference
    live_counts: dict = {}
    state.manual_scan_result = {"status": "running", "started_at": datetime.now(timezone.utc).isoformat(), "counts": live_counts}
    state.scan_low_conf_assets = []
    scan_since_iso = scan_since + "T00:00:00.000Z"

    def on_date(date_str):
        if isinstance(state.manual_scan_result, dict):
            state.manual_scan_result["current_date"] = date_str

    def on_counts(counts):
        live_counts.clear()
        live_counts.update(counts)

    try:
        async with state.scan_lock:
            if state.scan_generation != generation:
                return
            state.scan_cancel.clear()
            counts, low_conf_assets = await asyncio.to_thread(
                inference.run_scan, str(DATA_DIR),
                manual=True, scan_until=scan_until, scan_since=scan_since_iso,
                cancel=state.scan_cancel, on_date=on_date, on_counts=on_counts,
            )
            if state.scan_generation == generation:
                state.scan_low_conf_assets = low_conf_assets
                state.manual_scan_result = data.load_poll_status(DATA_DIR)
    except Exception as e:
        if state.scan_generation == generation:
            state.manual_scan_result = {"status": "error", "error": str(e), "ran_at": datetime.now(timezone.utc).isoformat()}


@router.get("/scan/result")
async def get_scan_result():
    result = state.manual_scan_result
    if not result:
        return {"status": "none"}
    skipped = set(data.load_skipped_ids(DATA_DIR))
    neg_index = data.index_crop_entries(data.load_negatives(DATA_DIR))
    filtered_count = len({
        a["asset_id"] for a in (state.scan_low_conf_assets or [])
        if a["asset_id"] not in skipped and not data.crop_entry_matches(neg_index, a["asset_id"], a.get("bbox"))
    })
    counts = {**result.get("counts", {}), "low_confidence": filtered_count}
    return {**result, "counts": counts}


@router.get("/scan/low-confidence")
async def get_scan_low_confidence():
    from poller import THRESHOLD
    config = data.load_config(DATA_DIR)
    skipped = set(data.load_skipped_ids(DATA_DIR))
    neg_index = data.index_crop_entries(data.load_negatives(DATA_DIR))
    seen: dict = {}
    for a in (state.scan_low_conf_assets or []):
        aid = a["asset_id"]
        if aid in skipped or data.crop_entry_matches(neg_index, aid, a.get("bbox")):
            continue
        if aid not in seen or a["prob"] > seen[aid]["prob"]:
            seen[aid] = a
    sorted_assets = sorted(seen.values(), key=lambda a: a["prob"])
    def make_item(a: dict) -> dict:
        aid = a["asset_id"]
        bbox = a.get("bbox")
        thumb = f"/api/crop/{aid}?bbox={','.join(str(v) for v in bbox)}" if bbox else f"/api/crop/{aid}"
        return {"id": aid, "thumb": thumb, "bbox": bbox,
                "pet_name": a["pet_name"], "score": a["prob"], "date": a.get("date", "")}

    return {
        "assets": [make_item(a) for a in sorted_assets],
        "pets": list(config.keys()),
        "threshold": THRESHOLD,
    }


# ---------------------------------------------------------------------------
# Immich people list (for import)
# ---------------------------------------------------------------------------

@router.get("/immich-people")
async def list_immich_people():
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(f"{imm.IMMICH_URL}/api/people", params={"withHidden": "false"}, headers=imm.headers())
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail="Failed to fetch people from Immich")
    body = resp.json()
    people = [{"id": p["id"], "name": p.get("name", "")} for p in body.get("people", []) if p.get("name")]
    return {"people": people}


# ---------------------------------------------------------------------------
# Thumbnail proxy
# ---------------------------------------------------------------------------

# Browser caching for proxied images. An asset's thumbnail and a crop of it never change,
# so grids re-rendered on every pet/list switch come from the browser cache instead of
# going through Immich again. Person thumbnails can change, so they get a short lifetime.
# Only successful responses are cacheable, so a failed fetch is retried next time.
ASSET_IMAGE_CACHE = {"Cache-Control": "private, max-age=604800, immutable"}
PERSON_IMAGE_CACHE = {"Cache-Control": "private, max-age=3600"}
CROP_TILE_MAX = 512  # max side in px of a crop sent to the UI


def _cache_if_ok(status_code: int, headers: dict) -> dict:
    return headers if status_code == 200 else {}


@router.get("/person-thumb/{person_id}")
async def person_thumbnail(person_id: str):
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(f"{imm.IMMICH_URL}/api/people/{person_id}/thumbnail", headers=imm.headers())
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code)
    return StreamingResponse(resp.aiter_bytes(), media_type=resp.headers.get("content-type", "image/jpeg"),
                             headers=PERSON_IMAGE_CACHE)


# ---------------------------------------------------------------------------
# Manual asset lookup (add a ref or negative by Immich link or ID)
# ---------------------------------------------------------------------------

@router.get("/asset/{asset_id}/crops")
async def get_asset_crops(asset_id: str):
    """Look up one asset by ID and return its detected animal crops, so a user can
    manually add a reference or negative by pasting an Immich photo link or ID."""
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(f"{imm.IMMICH_URL}/api/assets/{asset_id}", headers=imm.headers())
    if resp.status_code != 200:
        raise HTTPException(status_code=404, detail="Asset not found. Check the link or ID.")
    meta = resp.json()
    def lookup():
        with inference_session():
            return emb.get_crops_and_embed(asset_id)
    try:
        crops_embed = await asyncio.to_thread(lookup)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    crops = [c for c, _ in crops_embed]
    return {**_slim_asset(meta), "crops": crops}


async def _grid_thumbnail(asset_id: str) -> StreamingResponse:
    """Immich's small thumbnail (about 250px) for grid tiles. The large preview size is only
    needed as a source for crops and for embedding (embedder.fetch_thumbnail)."""
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(f"{imm.IMMICH_URL}/api/assets/{asset_id}/thumbnail?size=thumbnail", headers=imm.headers())
    if resp.status_code != 200:
        log.warning(f"Grid thumbnail {asset_id}: Immich returned {resp.status_code}")
    return StreamingResponse(resp.aiter_bytes(), status_code=resp.status_code,
                             media_type=resp.headers.get("content-type", "image/jpeg"),
                             headers=_cache_if_ok(resp.status_code, ASSET_IMAGE_CACHE))


@router.get("/thumb/{asset_id}")
async def thumbnail(asset_id: str):
    return await _grid_thumbnail(asset_id)


@router.get("/crop/{asset_id}")
async def animal_crop(asset_id: str, bbox: str | None = None):
    """Return a cropped animal region by bbox (x1,y1,x2,y2 normalized), or the full thumbnail."""
    if bbox:
        try:
            x1, y1, x2, y2 = [float(v) for v in bbox.split(",")]
            def do_crop():
                img = emb.fetch_thumbnail(asset_id)
                if img is None:
                    return None
                w, h = img.size
                crop = img.crop((int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)))
                crop.thumbnail((CROP_TILE_MAX, CROP_TILE_MAX))  # tiles are small, keep the download light
                return crop
            crop = await asyncio.to_thread(do_crop)
            if crop is not None:
                buf = io.BytesIO()
                crop.save(buf, "JPEG", quality=85)
                buf.seek(0)
                return StreamingResponse(buf, media_type="image/jpeg", headers=ASSET_IMAGE_CACHE)
        except Exception as e:
            log.warning(f"Crop {asset_id} bbox={bbox} failed, serving full thumbnail: {e}")
    return await _grid_thumbnail(asset_id)
