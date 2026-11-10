"""File I/O helpers. All functions take an explicit data_dir Path."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger("data")


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(data_dir: Path) -> dict:
    f = data_dir / "config.json"
    if not f.exists():
        return {}
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except Exception as e:
        log.error(f"Corrupted {f}, returning empty config: {e}")
        return {}


def save_config(config: dict, data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write(data_dir / "config.json", json.dumps(config, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------------------
# Pet refs
# ---------------------------------------------------------------------------

def load_pet_refs(pet_name: str, data_dir: Path) -> list[dict]:
    """Return list of {asset_id, face_id}. Handles legacy list-of-strings format."""
    ref_file = data_dir / "pets" / pet_name / "refs.json"
    if not ref_file.exists():
        return []
    try:
        data = json.loads(ref_file.read_text(encoding="utf-8"))
    except Exception as e:
        log.error(f"Corrupted {ref_file}, returning empty refs: {e}")
        return []
    if not data:
        return []
    if isinstance(data[0], str):
        return [{"asset_id": aid, "face_id": None} for aid in data]
    return data


def load_pet_asset_ids(pet_name: str, data_dir: Path) -> list[str]:
    seen: set[str] = set()
    result = []
    for r in load_pet_refs(pet_name, data_dir):
        aid = r["asset_id"]
        if aid not in seen:
            seen.add(aid)
            result.append(aid)
    return result


def save_pet_refs(pet_name: str, refs: list[dict], data_dir: Path) -> None:
    pet_dir = data_dir / "pets" / pet_name
    pet_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write(pet_dir / "refs.json", json.dumps(refs, indent=2))


# ---------------------------------------------------------------------------
# Crop entries: {"asset_id": str, "bbox": [x1, y1, x2, y2] | None}
# A bbox (normalized) marks one detected crop of the photo. bbox None means the
# whole photo, all its crops. A bare asset id string (legacy format) is read as
# a whole-photo entry.
# ---------------------------------------------------------------------------

BBOX_MATCH_TOLERANCE = 0.02  # max per-coordinate difference for two bboxes to be the same crop


def normalize_crop_entries(raw) -> list[dict]:
    out = []
    for item in raw or []:
        if isinstance(item, str):
            out.append({"asset_id": item, "bbox": None})
        elif isinstance(item, dict) and item.get("asset_id"):
            bbox = item.get("bbox")
            out.append({"asset_id": item["asset_id"], "bbox": [float(c) for c in bbox] if bbox else None})
    return out


def crop_entry_key(entry: dict) -> tuple:
    """Stable, hashable and sortable identity of an entry. Whole-photo entries get ()."""
    bbox = entry.get("bbox")
    return (entry["asset_id"], tuple(round(c, 4) for c in bbox) if bbox else ())


def crop_entry_token(entry: dict) -> str:
    """String form of crop_entry_key. A whole-photo entry is just its asset id."""
    aid, bbox = crop_entry_key(entry)
    return aid + ("@" + ",".join(f"{c:.4f}" for c in bbox) if bbox else "")


def merge_crop_entries(existing: list[dict], new: list[dict]) -> list[dict]:
    """existing followed by the new entries, deduped on crop_entry_key, order kept."""
    seen: set = set()
    out = []
    for e in list(existing) + list(new):
        k = crop_entry_key(e)
        if k not in seen:
            seen.add(k)
            out.append(e)
    return out


def bboxes_match(a, b, tol: float = BBOX_MATCH_TOLERANCE) -> bool:
    return bool(a) and bool(b) and len(a) == len(b) and all(abs(x - y) <= tol for x, y in zip(a, b))


def index_crop_entries(entries: list[dict]) -> dict[str, list]:
    """asset_id -> list of entry bboxes (None for a whole-photo entry)."""
    idx: dict[str, list] = {}
    for e in entries:
        idx.setdefault(e["asset_id"], []).append(e.get("bbox"))
    return idx


def whole_photo_ids(index: dict[str, list]) -> set[str]:
    """Assets with a whole-photo entry, i.e. excluded entirely."""
    return {aid for aid, bboxes in index.items() if None in bboxes}


def crop_entry_matches(index: dict[str, list], asset_id: str, bbox=None) -> bool:
    """True if the crop (asset_id, bbox) is covered: by a whole-photo entry of the asset,
    or by a crop entry whose bbox matches. A crop with no bbox (whole-image fallback)
    only matches a whole-photo entry."""
    bboxes = index.get(asset_id)
    if not bboxes:
        return False
    if None in bboxes:
        return True
    return bbox is not None and any(bboxes_match(bbox, b) for b in bboxes)


# ---------------------------------------------------------------------------
# Negatives (list of crop entries)
# ---------------------------------------------------------------------------

def load_negatives(data_dir: Path) -> list[dict]:
    path = data_dir / "negatives.json"
    if not path.exists():
        return []
    try:
        return normalize_crop_entries(json.loads(path.read_text(encoding="utf-8")))
    except Exception as e:
        log.error(f"Corrupted {path}, returning empty negatives: {e}")
        return []


def save_negatives(entries: list[dict], data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write(data_dir / "negatives.json", json.dumps(normalize_crop_entries(entries), indent=2))


# ---------------------------------------------------------------------------
# Skipped
# ---------------------------------------------------------------------------

def load_skipped_ids(data_dir: Path) -> list[str]:
    path = data_dir / "skipped.json"
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        log.error(f"Corrupted {path}, returning empty skipped list: {e}")
        return []


def save_skipped_ids(ids: list[str], data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write(data_dir / "skipped.json", json.dumps(ids, indent=2))


# ---------------------------------------------------------------------------
# Scan timestamp
# ---------------------------------------------------------------------------

def load_last_timestamp(data_dir: Path) -> str:
    path = data_dir / "last_scan_timestamp.txt"
    default = datetime.now(timezone.utc).date().isoformat() + "T00:00:00.000Z"
    if not path.exists():
        path.write_text(default + "\n", encoding="utf-8")
        return default
    val = path.read_text(encoding="utf-8").strip()
    return val if val else default


def save_last_timestamp(ts: str, data_dir: Path) -> None:
    _atomic_write(data_dir / "last_scan_timestamp.txt", ts.strip() + "\n")


# ---------------------------------------------------------------------------
# Poll status
# ---------------------------------------------------------------------------

def load_poll_status(data_dir: Path) -> dict:
    path = data_dir / "last_poll_status.json"
    if not path.exists():
        return {"status": "never"}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        log.error(f"Corrupted {path}, returning default status: {e}")
        return {"status": "never"}


def write_poll_status(data_dir: Path, payload: dict) -> None:
    try:
        _atomic_write(data_dir / "last_poll_status.json", json.dumps(payload))
    except Exception:
        pass
