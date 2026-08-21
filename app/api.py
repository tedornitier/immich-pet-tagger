"""API routes for the enrollment UI.
All Immich communication happens here; the browser never touches Immich directly."""

import asyncio
import hashlib
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
from inference import inference_session

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
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=KEEPALIVE_INTERVAL)
            except asyncio.TimeoutError:
                continue
        result = await task
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
    """Env-var-configured defaults (THRESHOLD, YOLO_CONF). Read-only: production config
    lives in docker-compose.yml only. Used to prefill the benchmark tool's per-run
    threshold overrides, which are never persisted, they only apply to that one run."""
    from poller import THRESHOLD
    return {"threshold": THRESHOLD, "yolo_conf": det.YOLO_CONF}


def _slim_asset(a: dict) -> dict:
    return {"id": a["id"], "thumb": f"/api/crop/{a['id']}", "date": a.get("localDateTime", "")[:10], "filename": a.get("originalFileName", "")}



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

    base: dict = {"type": "IMAGE", "size": per_ref_limit}
    if pet_cfg.get("since"):
        base["takenAfter"] = pet_cfg["since"] + "T00:00:00.000Z"
    if pet_cfg.get("until"):
        base["takenBefore"] = pet_cfg["until"] + "T23:59:59.999Z"

    async def fetch_one(rid: str) -> list[dict]:
        try:
            resp = await client.post(
                f"{imm.IMMICH_URL}/api/search/smart",
                headers=imm.headers(),
                json={**base, "queryAssetId": rid},
            )
            if resp.status_code == 200:
                return resp.json().get("assets", {}).get("items", [])
        except Exception:
            pass
        return []

    results = await asyncio.gather(*[fetch_one(rid) for rid in sampled])
    seen: set[str] = set()
    candidates: list[dict] = []
    for items in results:
        for a in items:
            aid = a.get("id")
            if aid and aid not in exclude and aid not in seen:
                seen.add(aid)
                candidates.append(a)
    return candidates


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
                await client.put(f"{imm.IMMICH_URL}/api/people/{person_id}", headers=imm.headers(), json={"name": new_name})
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

@router.get("/negatives")
async def get_negatives():
    ids = data.load_negative_ids(DATA_DIR)
    return {"assets": [{"id": aid, "thumb": f"/api/thumb/{aid}"} for aid in ids], "count": len(ids)}


@router.post("/negatives")
async def add_negatives(body: PetAssets):
    existing = set(data.load_negative_ids(DATA_DIR))
    merged = list(existing | set(body.asset_ids))
    data.save_negative_ids(merged, DATA_DIR)
    log.info(f"Negatives: {len(merged)} total (+{len(set(body.asset_ids) - existing)} new)")
    return {"ok": True, "count": len(merged)}


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
    data.save_negative_ids([], DATA_DIR)
    log.info("Cleared all negatives (local only)")
    return {"ok": True}


@router.delete("/negatives/{asset_id}")
async def remove_negative(asset_id: str):
    ids = [i for i in data.load_negative_ids(DATA_DIR) if i != asset_id]
    data.save_negative_ids(ids, DATA_DIR)
    return {"ok": True}


# ---------------------------------------------------------------------------
# Skipped
# ---------------------------------------------------------------------------

@router.post("/skipped")
async def add_skipped(body: PetAssets):
    existing = set(data.load_skipped_ids(DATA_DIR))
    merged = list(existing | set(body.asset_ids))
    data.save_skipped_ids(merged, DATA_DIR)
    return {"count": len(merged)}


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

def _classifier_fingerprint(pet_names: list[str], refs_per_pet: dict, negative_ids: list[str]) -> str:
    """Stable hash of the inputs that define a trained classifier."""
    parts = []
    for name in sorted(pet_names):
        parts.append(name + ":" + ",".join(sorted(r["asset_id"] for r in refs_per_pet[name])))
    parts.append("neg:" + ",".join(sorted(negative_ids)))
    return hashlib.md5("\n".join(parts).encode()).hexdigest()


def _build_classifier_from_config(config: dict):
    """Load all pet refs and return (names, clf, scaler), or None if no pets have refs.

    The trained classifier is cached in memory and only rebuilt when the set of
    refs or negatives changes. This avoids re-embedding every ref on each request
    and keeps prediction scores stable between calls."""
    from classifier import build_classifier
    all_pet_names = list(config.keys())
    all_refs = {n: data.load_pet_refs(config[n].get("person_id") or n, DATA_DIR) for n in all_pet_names}
    pet_names = [n for n in all_pet_names if all_refs.get(n)]
    refs_per_pet = {n: all_refs[n] for n in pet_names}
    negative_ids = data.load_negative_ids(DATA_DIR)

    fp = _classifier_fingerprint(pet_names, refs_per_pet, negative_ids)
    with state.classifier_cache_lock:
        cached = state.classifier_cache
        if cached is not None and cached["fingerprint"] == fp:
            return cached["names"], cached["clf"], cached["scaler"]

    result = build_classifier(pet_names, refs_per_pet, negative_ids)
    if result is None:
        return None
    names, clf, scaler = result
    with state.classifier_cache_lock:
        state.classifier_cache = {"fingerprint": fp, "names": names, "clf": clf, "scaler": scaler}
    return names, clf, scaler


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
    neg_ids = set(data.load_negative_ids(DATA_DIR))
    exclude = ref_set | neg_ids

    async with httpx.AsyncClient(timeout=30) as client:
        if ref_ids:
            candidates = await _visual_search(client, ref_ids, pet_cfg, exclude)
        else:
            body: dict = {"query": description, "type": "IMAGE", "size": 60}
            if pet_cfg.get("since"):
                body["takenAfter"] = pet_cfg["since"] + "T00:00:00.000Z"
            if pet_cfg.get("until"):
                body["takenBefore"] = pet_cfg["until"] + "T23:59:59.999Z"
            resp = await client.post(f"{imm.IMMICH_URL}/api/search/smart", headers=imm.headers(), json=body)
            if resp.status_code != 200:
                raise HTTPException(status_code=resp.status_code, detail=resp.text)
            all_items = resp.json().get("assets", {}).get("items", [])
            candidates = [a for a in all_items if a["id"] not in exclude]

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
    neg_ids = set(data.load_negative_ids(DATA_DIR))
    skipped_ids = set(data.load_skipped_ids(DATA_DIR))
    exclude = ref_set | neg_ids | skipped_ids

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
                result = _build_classifier_from_config(config)
                if result is None:
                    return []
                names, clf, scaler = result
                if name not in names:
                    return []
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
                            v = np.asarray(vec, dtype=np.float64).reshape(1, -1)
                            pet_prob = float(clf.predict_proba(scaler.transform(v))[0][pet_idx])
                            if LOW <= pet_prob < HIGH:
                                scored.append((pet_prob, {**_slim_asset(a), "crops": [c], "score": round(pet_prob, 3)}))
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
    threshold for every detection (see .claude/decisions.md)."""
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


@router.get("/suggestions/negatives")
async def get_neg_candidates(limit: int = 60):
    from poller import THRESHOLD
    config = data.load_config(DATA_DIR)

    all_pet_names = list(config.keys())
    all_refs = {n: data.load_pet_refs(config[n].get("person_id") or n, DATA_DIR) for n in all_pet_names}
    all_ref_ids: set[str] = {r["asset_id"] for refs in all_refs.values() for r in refs}
    neg_ids = set(data.load_negative_ids(DATA_DIR))
    skipped_ids = set(data.load_skipped_ids(DATA_DIR))
    exclude = all_ref_ids | neg_ids | skipped_ids

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{imm.IMMICH_URL}/api/search/random",
            headers=imm.headers(),
            json={"size": 50, "type": "IMAGE"},
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
                result = _build_classifier_from_config(config)
                if result is None:
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
            json={"personIds": [body.person_id], "size": 200},
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
    skipped = set(data.load_skipped_ids(DATA_DIR)) | set(data.load_negative_ids(DATA_DIR))
    # Recounted rather than taken from the scan, since assets can be marked skipped or
    # negative after the fact. Whole-image matches held back by WHOLE_IMAGE_MATCH=review
    # share the queue but are counted apart, so "Low conf." keeps meaning what it did.
    pending = [a for a in (state.scan_low_conf_assets or []) if a["asset_id"] not in skipped]
    counts = {
        **result.get("counts", {}),
        "low_confidence": len({a["asset_id"] for a in pending if a.get("outcome") != "whole_image_review"}),
        "whole_image_review": len({a["asset_id"] for a in pending if a.get("outcome") == "whole_image_review"}),
    }
    return {**result, "counts": counts}


@router.get("/scan/low-confidence")
async def get_scan_low_confidence():
    from poller import THRESHOLD
    config = data.load_config(DATA_DIR)
    skipped = set(data.load_skipped_ids(DATA_DIR)) | set(data.load_negative_ids(DATA_DIR))
    seen: dict = {}
    for a in (state.scan_low_conf_assets or []):
        aid = a["asset_id"]
        if aid in skipped:
            continue
        if aid not in seen or a["prob"] > seen[aid]["prob"]:
            seen[aid] = a
    sorted_assets = sorted(seen.values(), key=lambda a: a["prob"])
    def make_item(a: dict) -> dict:
        aid = a["asset_id"]
        bbox = a.get("bbox")
        thumb = f"/api/crop/{aid}?bbox={','.join(str(v) for v in bbox)}" if bbox else f"/api/crop/{aid}"
        # No bbox in this queue means YOLO found nothing and the match came from the
        # whole photo. Worth flagging in the grid: accepting one as a reference teaches
        # the classifier to match on the background instead of on the pet.
        return {"id": aid, "thumb": thumb, "bbox": bbox, "full_image": bbox is None,
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

@router.get("/person-thumb/{person_id}")
async def person_thumbnail(person_id: str):
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(f"{imm.IMMICH_URL}/api/people/{person_id}/thumbnail", headers=imm.headers())
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code)
    return StreamingResponse(resp.aiter_bytes(), media_type=resp.headers.get("content-type", "image/jpeg"))


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


@router.get("/thumb/{asset_id}")
async def thumbnail(asset_id: str):
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(f"{imm.IMMICH_URL}/api/assets/{asset_id}/thumbnail?size=preview", headers=imm.headers())
    return StreamingResponse(resp.aiter_bytes(), media_type=resp.headers.get("content-type", "image/jpeg"))


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
                return img.crop((int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)))
            crop = await asyncio.to_thread(do_crop)
            if crop is not None:
                buf = io.BytesIO()
                crop.save(buf, "JPEG", quality=85)
                buf.seek(0)
                return StreamingResponse(buf, media_type="image/jpeg")
        except Exception:
            pass
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(f"{imm.IMMICH_URL}/api/assets/{asset_id}/thumbnail?size=preview", headers=imm.headers())
    return StreamingResponse(resp.aiter_bytes(), media_type=resp.headers.get("content-type", "image/jpeg"))
