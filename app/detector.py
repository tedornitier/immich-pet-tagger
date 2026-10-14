"""Animal detector using YOLOv8n. Batched inference via queue, N parallel worker threads.
Pre-processing (PIL→tensor) happens in caller threads; batch threads only run the GPU kernel."""

import logging
import os
import queue
import threading
import time

import numpy as np
import torch
from PIL import Image

log = logging.getLogger("detector")

YOLO_BATCH_SIZE = int(os.environ.get("YOLO_BATCH_SIZE", 32))
YOLO_WORKERS = int(os.environ.get("GPU_WORKERS", 2))
YOLO_INPUT_SIZE = int(os.environ.get("YOLO_INPUT_SIZE", 640))
YOLO_MODEL_NAME = os.environ.get("YOLO_MODEL_NAME", "yolov8n.pt")
YOLO_CONF = float(os.environ.get("YOLO_CONF", 0.25))
IOU_THRESHOLD = float(os.environ.get("IOU_THRESHOLD", 0.7))
# Cross-class duplicate suppression, applied to animal boxes only (after the class
# filter, so a person holding a cat can never suppress the cat). YOLO's own NMS is
# per class: the same cat seen as "cat" 0.9 and "dog" 0.3 survives twice, and a
# box around just a leg or the tail survives inside the whole-animal box because
# IoU against a much larger box stays low. Each surviving box is then classified
# on its own, so one animal could be tagged as two different pets.
# A box is dropped when its IoU with a kept box exceeds IOU_THRESHOLD, or when more
# than CONTAINMENT_THRESHOLD of its own area lies inside a kept box. Set to 1.0 to
# disable the containment rule.
CONTAINMENT_THRESHOLD = float(os.environ.get("CONTAINMENT_THRESHOLD", 0.8))

# Passed to the model call itself: a permissive floor so boxes are never discarded before
# reaching Python. The real cutoff (YOLO_CONF, optionally overridden per request, e.g. by
# the benchmark tool) is applied afterward, so testing a different confidence never needs
# a second inference pass over the same image.
_MODEL_CONF_FLOOR = 0.001

ANIMAL_CLASS_IDS = {
    14,  # bird
    15,  # cat
    16,  # dog
    17,  # horse
    18,  # sheep
    19,  # cow
    20,  # elephant
    21,  # bear
    22,  # zebra
    23,  # giraffe
}


class _YoloReq:
    __slots__ = ("tensor", "conf", "event", "result")
    def __init__(self, tensor: torch.Tensor, conf: float):
        self.tensor = tensor
        self.conf = conf
        self.event = threading.Event()
        self.result: list | None = None


_YOLO_STOP = object()

_yolo_queue: queue.Queue = queue.Queue()
_yolo_worker_threads: list[threading.Thread] = []
_yolo_worker_lock = threading.Lock()

yolo_batch_total = 0
yolo_batch_count = 0
_yolo_stats_lock = threading.Lock()

# Set when the first YOLO worker finishes loading. Never set if loading fails.
_yolo_worker_ready = threading.Event()
# Set to an error string if loading fails.
_yolo_load_error: str | None = None
# Batches that failed at runtime, see the same pair in embedder.
runtime_errors = 0
last_runtime_error: tuple[float, str] | None = None  # (time.monotonic(), message)


def is_yolo_ready() -> bool:
    return _yolo_worker_ready.is_set()


def get_yolo_error() -> str | None:
    return _yolo_load_error


def _yolo_batch_loop(worker_id: int) -> None:
    global yolo_batch_total, yolo_batch_count, _yolo_load_error, runtime_errors, last_runtime_error
    from ultralytics import YOLO
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"YOLO worker {worker_id} loading on {device}...")
    try:
        model = YOLO(YOLO_MODEL_NAME)
        model.to(device)
    except Exception as e:
        _yolo_load_error = str(e)
        log.error(
            f"YOLO worker {worker_id} failed to load: {e}. "
            "On first start the model is downloaded (~6 MB). "
            "Ensure the container has internet access, then restart. "
            "Alternatively, copy yolov8n.pt into the data volume manually."
        )
        return
    _yolo_worker_ready.set()
    log.info(f"YOLO worker {worker_id} ready")

    while True:
        first = _yolo_queue.get()
        if first is _YOLO_STOP:
            break
        batch = [first]
        try:
            while len(batch) < YOLO_BATCH_SIZE:
                batch.append(_yolo_queue.get_nowait())
        except queue.Empty:
            pass

        with _yolo_stats_lock:
            yolo_batch_total += len(batch)
            yolo_batch_count += 1

        try:
            # Tensors are already preprocessed by caller threads: B×C×H×W, float32, [0,1], RGB.
            # Ultralytics skips PIL/numpy conversion when given a tensor directly.
            stacked = torch.stack([req.tensor for req in batch])
            results_list = model(stacked, verbose=False, imgsz=YOLO_INPUT_SIZE, conf=_MODEL_CONF_FLOOR, iou=IOU_THRESHOLD)
            for req, result in zip(batch, results_list):
                boxes = []
                for box in result.boxes:
                    cls = int(box.cls[0])
                    if cls not in ANIMAL_CLASS_IDS:
                        continue
                    conf = float(box.conf[0])
                    if conf < req.conf:
                        continue
                    x1, y1, x2, y2 = box.xyxyn[0].tolist()
                    boxes.append((conf, x1, y1, x2, y2))
                boxes.sort(reverse=True)
                req.result = dedupe_boxes(boxes)  # (conf, x1, y1, x2, y2), highest confidence first
                req.event.set()
        except Exception as e:
            log.warning(f"YOLO worker {worker_id} batch error: {e}")
            with _yolo_stats_lock:
                runtime_errors += 1
                last_runtime_error = (time.monotonic(), str(e) or type(e).__name__)
            for req in batch:
                req.result = []
                req.event.set()


def dedupe_boxes(boxes: list[tuple[float, float, float, float, float]]) -> list[tuple[float, float, float, float, float]]:
    """Merge animal boxes that are the same animal, across classes. Input and output are
    (conf, x1, y1, x2, y2), highest confidence first.

    1. Cross-class duplicates: a box whose IoU with a more confident kept box exceeds
       IOU_THRESHOLD is dropped.
    2. Group boxes: a box that contains two or more other boxes (each more than
       CONTAINMENT_THRESHOLD inside it) is dropped and the inner boxes kept. It is either
       a box around two pets lying together, or one pet split in parts; geometry cannot
       tell those apart, and merging two pets into one box loses one of them.
    3. Whole animal vs one part of it: when one box lies inside another, the outer box
       (the whole animal) is kept if its confidence is at least half the inner one's,
       otherwise the inner box is kept (a large, unsure box does not swallow a confident
       detection). Two boxes each inside the other keep the more confident one."""
    def area(b):
        return max(0.0, b[3] - b[1]) * max(0.0, b[4] - b[2])

    def inter(a, b):
        ix = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
        iy = max(0.0, min(a[4], b[4]) - max(a[2], b[2]))
        return ix * iy

    def inside(small, big):
        a = area(small)
        return a > 0 and inter(small, big) / a > CONTAINMENT_THRESHOLD

    kept: list[tuple[float, float, float, float, float]] = []
    for b in sorted(boxes, reverse=True):
        dup = False
        for k in kept:
            ov = inter(b, k)
            union = area(b) + area(k) - ov
            if union > 0 and ov / union > IOU_THRESHOLD:
                dup = True
                break
        if not dup:
            kept.append(b)

    groups = {
        i for i, big in enumerate(kept)
        if sum(1 for j, small in enumerate(kept) if j != i and inside(small, big)) >= 2
    }
    kept = [b for i, b in enumerate(kept) if i not in groups]

    drop: set[int] = set()
    for i, big in enumerate(kept):
        for j, small in enumerate(kept):
            if i == j or not inside(small, big):
                continue
            if inside(big, small):
                drop.add(j if big[0] >= small[0] else i)
            elif big[0] >= 0.5 * small[0]:
                drop.add(j)
            else:
                drop.add(i)
    out = [b for i, b in enumerate(kept) if i not in drop]
    out.sort(reverse=True)
    return out


def _ensure_yolo_workers() -> None:
    with _yolo_worker_lock:
        alive = [t for t in _yolo_worker_threads if t.is_alive()]
        for i in range(len(alive), YOLO_WORKERS):
            t = threading.Thread(target=_yolo_batch_loop, args=(i,), daemon=True, name=f"yolo-batch-{i}")
            t.start()
            _yolo_worker_threads.append(t)


def start_workers() -> None:
    _ensure_yolo_workers()


def stop_workers() -> None:
    global _yolo_load_error
    with _yolo_worker_lock:
        alive = [t for t in _yolo_worker_threads if t.is_alive()]
        for _ in alive:
            _yolo_queue.put(_YOLO_STOP)
        for t in alive:
            t.join(timeout=60)
        _yolo_worker_threads.clear()
        _yolo_worker_ready.clear()
        _yolo_load_error = None
    log.info("YOLO workers stopped")


def wait_for_ready(timeout: float = 300) -> None:
    _wait_for_yolo_ready(timeout)


def _wait_for_yolo_ready(timeout: float = 300) -> None:
    deadline = time.time() + timeout
    while not _yolo_worker_ready.is_set():
        if _yolo_load_error:
            raise RuntimeError(f"YOLO not available: {_yolo_load_error}")
        if time.time() > deadline:
            raise RuntimeError(_yolo_load_error or "YOLO worker did not become ready")
        time.sleep(0.1)


def detect_animals(img: Image.Image, conf: float | None = None) -> list[tuple[float, float, float, float, float]]:
    """Returns (conf, x1, y1, x2, y2) for detected animals, highest confidence first.
    conf overrides YOLO_CONF for this call only (e.g. a benchmark testing a different
    cutoff); the underlying inference is unaffected either way, only the filter is."""
    _ensure_yolo_workers()
    _wait_for_yolo_ready()
    # Pre-process in caller's thread (parallel across all scan workers).
    small = img.resize((YOLO_INPUT_SIZE, YOLO_INPUT_SIZE), Image.BILINEAR)
    arr = np.array(small, dtype=np.float32) / 255.0  # H×W×3, RGB, [0,1]
    tensor = torch.from_numpy(arr.transpose(2, 0, 1))  # C×H×W
    req = _YoloReq(tensor, conf if conf is not None else YOLO_CONF)
    _yolo_queue.put(req)
    if not req.event.wait(timeout=120):
        raise RuntimeError("YOLO worker did not respond within 120 s. Model may still be downloading.")
    return req.result
