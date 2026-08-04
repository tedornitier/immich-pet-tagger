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
# Ultralytics defaults this to 0.25, which is tuned for "is this detection
# trustworthy on its own". Here a detection only has to be good enough to crop
# on -- CLIP and the classifier decide what the animal actually is -- so a low
# threshold is wanted. At 0.25, pets that are small in frame, turned away,
# occluded, blurred or in low light are missed entirely.
YOLO_CONF = float(os.environ.get("YOLO_CONF", 0.10))

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
    __slots__ = ("tensor", "event", "result")
    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor
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


def is_yolo_ready() -> bool:
    return _yolo_worker_ready.is_set()


def get_yolo_error() -> str | None:
    return _yolo_load_error


def _yolo_batch_loop(worker_id: int) -> None:
    global yolo_batch_total, yolo_batch_count, _yolo_load_error
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
            results_list = model(stacked, verbose=False, imgsz=YOLO_INPUT_SIZE, conf=YOLO_CONF)
            for req, result in zip(batch, results_list):
                boxes = []
                for box in result.boxes:
                    cls = int(box.cls[0])
                    if cls not in ANIMAL_CLASS_IDS:
                        continue
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxyn[0].tolist()
                    boxes.append((conf, x1, y1, x2, y2))
                boxes.sort(reverse=True)
                req.result = [(x1, y1, x2, y2) for _, x1, y1, x2, y2 in boxes]
                req.event.set()
        except Exception as e:
            log.warning(f"YOLO worker {worker_id} batch error: {e}")
            for req in batch:
                req.result = []
                req.event.set()


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


def detect_animals(img: Image.Image) -> list[tuple[float, float, float, float]]:
    """Returns (x1, y1, x2, y2) normalized bboxes for detected animals, sorted by confidence."""
    _ensure_yolo_workers()
    _wait_for_yolo_ready()
    # Pre-process in caller's thread (parallel across all scan workers).
    small = img.resize((YOLO_INPUT_SIZE, YOLO_INPUT_SIZE), Image.BILINEAR)
    arr = np.array(small, dtype=np.float32) / 255.0  # H×W×3, RGB, [0,1]
    tensor = torch.from_numpy(arr.transpose(2, 0, 1))  # C×H×W
    req = _YoloReq(tensor)
    _yolo_queue.put(req)
    if not req.event.wait(timeout=120):
        raise RuntimeError("YOLO worker did not respond within 120 s. Model may still be downloading.")
    return req.result
