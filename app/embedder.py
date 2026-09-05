"""CLIP batch inference workers and image embedding."""

import io
import logging
import os
import pickle
import queue
import re
import sqlite3
import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np
import open_clip
import requests
import torch
from PIL import Image

import immich as imm
from device import is_gpu, pick_device, submit_lock

log = logging.getLogger("embedder")

GPU_WORKERS = int(os.environ.get("GPU_WORKERS", 2))
_default_scan_workers = GPU_WORKERS * 32 if is_gpu(pick_device()) else 8
SCAN_WORKERS = int(os.environ.get("SCAN_WORKERS", _default_scan_workers))
CLIP_BATCH_SIZE = int(os.environ.get("CLIP_BATCH_SIZE", 32))
CLIP_MODEL_NAME = os.environ.get("CLIP_MODEL_NAME", "ViT-B-16")
CLIP_PRETRAINED = os.environ.get("CLIP_PRETRAINED", "openai")
_DEFAULT_CLIP_MODEL_NAME = "ViT-B-16"
_DEFAULT_CLIP_PRETRAINED = "openai"
_DEFAULT_YOLO_MODEL_NAME = "yolov8n.pt"

MAX_EMBED_CACHE_SIZE = int(os.environ.get("EMBED_CACHE_SIZE", 5000))
_embed_cache: OrderedDict[str, list[np.ndarray]] = OrderedDict()
_cache_path: Path | None = None
_cache_dirty = False
_cache_lock = threading.Lock()

# Persistent crop cache: asset_id -> list of (bbox, vec) pairs, one per detected
# animal crop. An empty list means "no animal here". Keyed only by asset_id:
# embeddings depend only on the image, so entries are immutable and never
# invalidated when pets or date ranges change. Populated lazily by whatever
# touches an asset (poller, borderline, suggestions). Backed by SQLite so it
# grows on disk only with the assets we actually process, not the whole library.
_crops_db: sqlite3.Connection | None = None

# ---------------------------------------------------------------------------
# CLIP batch workers
# ---------------------------------------------------------------------------

# Preprocess transform shared across worker threads (set by first CLIP worker).
# Worker threads do CPU preprocessing; batch threads only stack + run GPU.
_clip_preprocess_fn = None
_clip_preprocess_ready = threading.Event()


class _EmbedReq:
    __slots__ = ("tensor", "event", "result")

    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor
        self.event = threading.Event()
        self.result: np.ndarray | None = None


_CLIP_STOP = object()

_embed_queue: queue.Queue = queue.Queue()
_clip_worker_threads: list[threading.Thread] = []
_clip_worker_lock = threading.Lock()

_clip_batch_total = 0
_clip_batch_count = 0
_stats_lock = threading.Lock()
_clip_load_error: str | None = None


def reset_batch_stats() -> None:
    global _clip_batch_total, _clip_batch_count
    with _stats_lock:
        _clip_batch_total = _clip_batch_count = 0


def get_avg_batch_size() -> float:
    with _stats_lock:
        return _clip_batch_total / _clip_batch_count if _clip_batch_count else 0.0


def is_clip_ready() -> bool:
    return _clip_preprocess_ready.is_set()


def get_clip_error() -> str | None:
    return _clip_load_error


def _clip_batch_loop(worker_id: int) -> None:
    global _clip_batch_total, _clip_batch_count, _clip_preprocess_fn, _clip_load_error
    device = pick_device()
    log.info(f"CLIP worker {worker_id} loading on {device}...")
    try:
        model, preprocess, _ = open_clip.create_model_and_transforms(CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED)
        with submit_lock(device):
            model.eval().to(device)
    except Exception as e:
        _clip_load_error = str(e)
        log.error(
            f"CLIP worker {worker_id} failed to load: {e}. "
            "On first start the CLIP model is downloaded (~350 MB). "
            "Ensure the container has internet access, then restart."
        )
        return
    if not _clip_preprocess_ready.is_set():
        _clip_preprocess_fn = preprocess
        _clip_preprocess_ready.set()
    stream = torch.cuda.Stream() if device == "cuda" else None
    log.info(f"CLIP worker {worker_id} ready")

    while True:
        first = _embed_queue.get()
        if first is _CLIP_STOP:
            break
        batch = [first]
        try:
            while len(batch) < CLIP_BATCH_SIZE:
                batch.append(_embed_queue.get_nowait())
        except queue.Empty:
            pass

        with _stats_lock:
            _clip_batch_total += len(batch)
            _clip_batch_count += 1

        try:
            stacked = torch.stack([req.tensor for req in batch])
            if stream is not None:
                with torch.cuda.stream(stream):
                    tensors = stacked.to(device, non_blocking=True)
                    with torch.no_grad():
                        feats = model.encode_image(tensors)
                        feats = feats / feats.norm(dim=-1, keepdim=True)
                stream.synchronize()
                vecs = feats.cpu().numpy()
            else:
                with submit_lock(device):
                    with torch.no_grad():
                        feats = model.encode_image(stacked.to(device))
                        feats = feats / feats.norm(dim=-1, keepdim=True)
                    vecs = feats.cpu().numpy()
        except Exception as e:
            log.warning(f"CLIP worker {worker_id} batch error: {e}")
            vecs = [None] * len(batch)

        for req, vec in zip(batch, vecs):
            req.result = vec
            req.event.set()


def _ensure_clip_workers() -> None:
    with _clip_worker_lock:
        alive = [t for t in _clip_worker_threads if t.is_alive()]
        for i in range(len(alive), GPU_WORKERS):
            t = threading.Thread(target=_clip_batch_loop, args=(i,), daemon=True, name=f"clip-batch-{i}")
            t.start()
            _clip_worker_threads.append(t)


def start_workers() -> None:
    _ensure_clip_workers()


def stop_workers() -> None:
    global _clip_preprocess_fn, _clip_load_error
    with _clip_worker_lock:
        alive = [t for t in _clip_worker_threads if t.is_alive()]
        for _ in alive:
            _embed_queue.put(_CLIP_STOP)
        for t in alive:
            t.join(timeout=60)
        _clip_worker_threads.clear()
        _clip_preprocess_ready.clear()
        _clip_preprocess_fn = None
        _clip_load_error = None
    log.info("CLIP workers stopped")


def wait_for_ready(timeout: float = 300) -> None:
    if not _clip_preprocess_ready.wait(timeout=timeout):
        raise RuntimeError(_clip_load_error or "CLIP worker did not become ready")
    if _clip_load_error:
        raise RuntimeError(f"CLIP not available: {_clip_load_error}")


# ---------------------------------------------------------------------------
# Thumbnail fetch and CLIP embedding
# ---------------------------------------------------------------------------

def fetch_thumbnail(asset_id: str) -> Image.Image | None:
    try:
        r = requests.get(
            f"{imm.IMMICH_URL}/api/assets/{asset_id}/thumbnail?size=preview",
            headers={"x-api-key": imm.IMMICH_API_KEY},
            timeout=30,
        )
        if r.status_code == 200 and r.content:
            return Image.open(io.BytesIO(r.content)).convert("RGB")
    except Exception as e:
        log.warning(f"fetch_thumbnail {asset_id}: {e}")
    return None


def embed_image(img: Image.Image) -> np.ndarray | None:
    _ensure_clip_workers()
    if not _clip_preprocess_ready.wait(timeout=300):  # 300 s covers a slow first-start download
        raise RuntimeError("CLIP worker did not respond within 300 s. Model may still be downloading.")
    tensor = _clip_preprocess_fn(img)  # CPU preprocessing in caller's thread
    req = _EmbedReq(tensor)
    _embed_queue.put(req)
    if not req.event.wait(timeout=120):
        raise RuntimeError("CLIP worker did not respond within 120 s.")
    return req.result


def crop_animals(img: Image.Image, conf: float | None = None, with_conf: bool = False):
    """Detect animals and return (bbox_norm, crop) pairs, or (bbox_norm, crop, detection_conf)
    triples if with_conf=True (e.g. to bucket by detection strength, not just presence).
    Empty list means no animals found. conf overrides YOLO_CONF for this call only, see
    detector.detect_animals."""
    try:
        from detector import detect_animals
        detections = detect_animals(img, conf=conf)  # (conf, x1, y1, x2, y2)
    except Exception as e:
        log.warning(f"YOLO detection failed: {e}")
        return []
    w, h = img.size
    out = []
    for det_conf, x1, y1, x2, y2 in detections:
        bbox = (x1, y1, x2, y2)
        crop = img.crop((int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)))
        out.append((bbox, crop, det_conf) if with_conf else (bbox, crop))
    return out


# ---------------------------------------------------------------------------
# Benchmark-only cache: raw thumbnail bytes + unfiltered YOLO detections.
# Deliberately separate from crops.db, never read or written by production code
# (poller, borderline, suggestions). crops.db is keyed only by asset_id and
# assumes production's fixed YOLO_CONF; letting a benchmark run's experimental
# confidence write there would silently corrupt what real tagging relies on.
# Bounded in-memory LRUs, no disk persistence: this is ephemeral diagnostic data
# for one session's benchmark iteration, not something worth surviving a restart.
# ---------------------------------------------------------------------------

BENCHMARK_CACHE_SIZE = int(os.environ.get("BENCHMARK_CACHE_SIZE", 20000))
_benchmark_thumb_cache: OrderedDict[str, bytes] = OrderedDict()
_benchmark_boxes_cache: OrderedDict[str, list] = OrderedDict()
_benchmark_cache_lock = threading.Lock()


def _lru_get(cache: OrderedDict, key):
    with _benchmark_cache_lock:
        val = cache.get(key)
        if val is not None:
            cache.move_to_end(key)
        return val


def _lru_put(cache: OrderedDict, key, val) -> None:
    with _benchmark_cache_lock:
        cache[key] = val
        cache.move_to_end(key)
        if len(cache) > BENCHMARK_CACHE_SIZE:
            cache.popitem(last=False)


def _benchmark_fetch_thumbnail(asset_id: str) -> Image.Image | None:
    cached = _lru_get(_benchmark_thumb_cache, asset_id)
    if cached is not None:
        return Image.open(io.BytesIO(cached)).convert("RGB")
    try:
        r = requests.get(
            f"{imm.IMMICH_URL}/api/assets/{asset_id}/thumbnail?size=preview",
            headers={"x-api-key": imm.IMMICH_API_KEY},
            timeout=30,
        )
        if r.status_code == 200 and r.content:
            _lru_put(_benchmark_thumb_cache, asset_id, r.content)
            return Image.open(io.BytesIO(r.content)).convert("RGB")
    except Exception as e:
        log.warning(f"fetch_thumbnail {asset_id}: {e}")
    return None


def _benchmark_raw_boxes(asset_id: str, img: Image.Image) -> list:
    cached = _lru_get(_benchmark_boxes_cache, asset_id)
    if cached is not None:
        return cached
    from detector import _MODEL_CONF_FLOOR, detect_animals
    boxes = detect_animals(img, conf=_MODEL_CONF_FLOOR)  # every detection YOLO produced
    _lru_put(_benchmark_boxes_cache, asset_id, boxes)
    return boxes


def crop_animals_cached(asset_id: str, conf: float | None = None, with_conf: bool = False):
    """Benchmark-only equivalent of crop_animals(): caches the thumbnail and the raw,
    unfiltered detections per asset_id (see the cache block above), so re-running the
    benchmark over the same assets with a different `conf` (e.g. comparing several YOLO
    thresholds) skips the Immich fetch and the YOLO pass entirely, only CLIP embedding
    of whichever crop the new confidence selects still runs fresh. Returns (img, crops):
    img is None if the thumbnail could not be fetched (crops is then always [])."""
    img = _benchmark_fetch_thumbnail(asset_id)
    if img is None:
        return None, []
    from detector import YOLO_CONF
    effective_conf = conf if conf is not None else YOLO_CONF
    boxes = sorted((b for b in _benchmark_raw_boxes(asset_id, img) if b[0] >= effective_conf), reverse=True)
    w, h = img.size
    out = []
    for det_conf, x1, y1, x2, y2 in boxes:
        bbox = (x1, y1, x2, y2)
        crop = img.crop((int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)))
        out.append((bbox, crop, det_conf) if with_conf else (bbox, crop))
    return img, out


def _crops_to_result(pairs: list[tuple[list, np.ndarray]]) -> list[tuple[dict, np.ndarray]]:
    return [({"crop_idx": i, "bbox": list(bbox)}, vec) for i, (bbox, vec) in enumerate(pairs)]


def store_crops(asset_id: str, pairs: list[tuple[list, np.ndarray]]) -> None:
    """Persist an asset's (bbox, vec) crop pairs. An empty list is stored too, marking
    the asset as having no detectable animal so we never reprocess it. Keyed only by
    asset_id and never invalidated, since embeddings depend only on the image."""
    if _crops_db is None:
        return
    try:
        blob = pickle.dumps(pairs)
        with _cache_lock:
            _crops_db.execute(
                "INSERT INTO crops (asset_id, data) VALUES (?, ?) "
                "ON CONFLICT(asset_id) DO UPDATE SET data = excluded.data",
                (asset_id, blob),
            )
            _crops_db.commit()
    except Exception as e:
        log.warning(f"Could not store crops for {asset_id}: {e}")


def _load_crops(asset_id: str) -> list[tuple[list, np.ndarray]] | None:
    if _crops_db is None:
        return None
    try:
        with _cache_lock:
            row = _crops_db.execute("SELECT data FROM crops WHERE asset_id = ?", (asset_id,)).fetchone()
        return pickle.loads(row[0]) if row is not None else None
    except Exception as e:
        log.warning(f"Could not read crops for {asset_id}: {e}")
        return None


def get_crops_and_embed(asset_id: str) -> list[tuple[dict, np.ndarray]]:
    """Fetch thumbnail once, run YOLO, embed each crop. Returns [(crop_info, vec), ...].
    Cached per asset in crops.db and reused across all pets and requests."""
    cached = _load_crops(asset_id)
    if cached is not None:
        return _crops_to_result(cached)
    img = fetch_thumbnail(asset_id)
    if img is None:
        return []  # transient fetch failure, do not cache
    pairs: list[tuple[list, np.ndarray]] = []
    for bbox, crop_img in crop_animals(img):
        vec = embed_image(crop_img)
        if vec is not None:
            pairs.append((list(bbox), vec))
    store_crops(asset_id, pairs)
    return _crops_to_result(pairs)


def embed_crop_by_bbox(asset_id: str, bbox: list) -> np.ndarray | None:
    """Embed a specific crop by normalized bounding box. Used for crop-centric refs."""
    global _cache_dirty
    with _cache_lock:
        cached = _embed_cache.get(asset_id)
        if cached is not None:
            _embed_cache.move_to_end(asset_id)
    if cached is not None:
        vecs = cached if isinstance(cached, list) else [cached]
        if len(vecs) == 1:
            return vecs[0]
    img = fetch_thumbnail(asset_id)
    if img is None:
        return None
    w, h = img.size
    x1, y1, x2, y2 = bbox
    crop_img = img.crop((int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)))
    vec = embed_image(crop_img)
    if vec is not None:
        with _cache_lock:
            _embed_cache[asset_id] = vec
            _embed_cache.move_to_end(asset_id)
            if len(_embed_cache) > MAX_EMBED_CACHE_SIZE:
                _embed_cache.popitem(last=False)
            _cache_dirty = True
    return vec


def _cache_suffix() -> str:
    """Cache files are namespaced by model so switching CLIP_MODEL_NAME/CLIP_PRETRAINED or
    YOLO_MODEL_NAME can't silently mix embeddings or crop boxes from different vector spaces
    or detectors. embeddings.pkl and crops.db both store crops of whatever YOLO detects, then
    CLIP-embeds them, so both models are part of the cache key. The default combo keeps the
    original unsuffixed filenames so existing installs don't lose their cache."""
    import detector as _det
    if (
        CLIP_MODEL_NAME == _DEFAULT_CLIP_MODEL_NAME
        and CLIP_PRETRAINED == _DEFAULT_CLIP_PRETRAINED
        and _det.YOLO_MODEL_NAME == _DEFAULT_YOLO_MODEL_NAME
    ):
        return ""
    raw = f"{CLIP_MODEL_NAME}_{CLIP_PRETRAINED}_{_det.YOLO_MODEL_NAME}"
    safe = re.sub(r"[^A-Za-z0-9]+", "_", raw).strip("_")[:80]
    return f"_{safe}"


def load_embed_cache(data_dir: Path) -> None:
    global _cache_path, _crops_db
    suffix = _cache_suffix()
    _cache_path = data_dir / f"embeddings{suffix}.pkl"
    if _cache_path.exists():
        try:
            with open(_cache_path, "rb") as f:
                loaded = pickle.load(f)
            with _cache_lock:
                _embed_cache.update(loaded)
                while len(_embed_cache) > MAX_EMBED_CACHE_SIZE:
                    _embed_cache.popitem(last=False)
            log.info(f"Loaded {len(_embed_cache)} cached embeddings from {_cache_path}")
        except Exception as e:
            log.warning(f"Could not load embedding cache: {e}")

    crops_db_path = data_dir / f"crops{suffix}.db"
    try:
        _crops_db = sqlite3.connect(crops_db_path, check_same_thread=False)
        _crops_db.execute("PRAGMA journal_mode=WAL")
        _crops_db.execute("PRAGMA synchronous=NORMAL")
        _crops_db.execute("CREATE TABLE IF NOT EXISTS crops (asset_id TEXT PRIMARY KEY, data BLOB NOT NULL)")
        _crops_db.commit()
        count = _crops_db.execute("SELECT COUNT(*) FROM crops").fetchone()[0]
        log.info(f"Crop cache ready: {count} assets in {crops_db_path}")
    except Exception as e:
        log.warning(f"Could not open crops cache db: {e}")
        _crops_db = None


def _save_embed_cache() -> None:
    global _cache_dirty
    if _cache_path is None:
        return
    with _cache_lock:
        if not _cache_dirty:
            return
        snapshot = dict(_embed_cache)
        _cache_dirty = False
    tmp = _cache_path.with_suffix(".tmp")
    try:
        with open(tmp, "wb") as f:
            pickle.dump(snapshot, f)
        tmp.replace(_cache_path)
    except Exception as e:
        log.warning(f"Could not save embedding cache: {e}")


def save_embed_cache() -> None:
    """Flush the embedding cache to disk. Call at the end of each scan cycle."""
    _save_embed_cache()


def embed_asset_crops(asset_id: str, require_animal: bool = False) -> list[np.ndarray]:
    """Return one embedding per detected animal crop. Falls back to full image if no crops and require_animal is False."""
    global _cache_dirty
    with _cache_lock:
        cached = _embed_cache.get(asset_id)
        if cached is not None:
            _embed_cache.move_to_end(asset_id)
    if cached is not None:
        return cached if isinstance(cached, list) else [cached]
    img = fetch_thumbnail(asset_id)
    if img is None:
        return []
    crops = crop_animals(img)
    if not crops:
        if require_animal:
            return []
        vec = embed_image(img)
        vecs = [vec] if vec is not None else []
    else:
        vecs = [v for v in (embed_image(crop_img) for _, crop_img in crops) if v is not None]
    if vecs:
        with _cache_lock:
            _embed_cache[asset_id] = vecs
            _embed_cache.move_to_end(asset_id)
            if len(_embed_cache) > MAX_EMBED_CACHE_SIZE:
                _embed_cache.popitem(last=False)
            _cache_dirty = True
    return vecs


def embed_asset(asset_id: str, require_animal: bool = False) -> np.ndarray | None:
    vecs = embed_asset_crops(asset_id, require_animal)
    return vecs[0] if vecs else None


def resolve_bbox(asset_id: str) -> list | None:
    """Return the first YOLO bounding box for an asset, or None if no animal detected."""
    img = fetch_thumbnail(asset_id)
    if img is None:
        return None
    crops = crop_animals(img)
    return list(crops[0][0]) if crops else None
