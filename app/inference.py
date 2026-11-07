"""On-demand YOLO/CLIP worker lifecycle, and subprocess-isolated scan execution.

Two different memory strategies for two different usage patterns:

- Interactive UI actions (suggestions, borderline, negatives, import) are
  short sessions where a human is actively working, so `inference_session()`
  loads workers in-process on first use and unloads them when the last caller
  exits. `torch.cuda.empty_cache()` reliably frees GPU VRAM; `malloc_trim(0)`
  is needed afterward too, since glibc's allocator otherwise keeps freed
  heap arenas mapped rather than returning them to the OS.
- Scans (background poll cycle and manual "Scan Now") run unattended, on a
  schedule, forever. `run_scan()` runs the whole cycle in a child OS process
  via scan_worker.py, so when it exits the OS reclaims all RAM/VRAM it used,
  keeping the web server's own footprint near zero between scans.
"""

import ctypes
import gc
import json
import logging
import subprocess
import sys
import threading
from contextlib import contextmanager
from pathlib import Path

import detector as det
import embedder as emb
from device import empty_cache, pick_device

log = logging.getLogger("inference")

try:
    _libc = ctypes.CDLL("libc.so.6")
except OSError:
    _libc = None

_lock = threading.Lock()
_refcount = 0


def is_active() -> bool:
    with _lock:
        return _refcount > 0


@contextmanager
def inference_session():
    """Load inference workers on first use, unload when the last caller exits."""
    global _refcount
    with _lock:
        if _refcount == 0:
            log.info("Loading inference models...")
            try:
                det.start_workers()
                emb.start_workers()
                det.wait_for_ready()
                emb.wait_for_ready()
            except Exception:
                det.stop_workers()
                emb.stop_workers()
                raise
            log.info("Inference models ready")
        _refcount += 1
    try:
        yield
    finally:
        with _lock:
            _refcount -= 1
            if _refcount == 0:
                log.info("Unloading inference models...")
                det.stop_workers()
                emb.stop_workers()
                gc.collect()
                empty_cache(pick_device())
                if _libc is not None:
                    _libc.malloc_trim(0)
                log.info("Inference models unloaded")


def model_error_count() -> int:
    """YOLO + CLIP batches that failed at runtime in this process. Compare before and
    after a piece of work to tell whether it hit a model error."""
    return det.runtime_errors + emb.runtime_errors


def describe_model_failure(since: int = 0) -> str | None:
    """The most recent runtime model error as one user-facing sentence, or None if no
    error was recorded after model_error_count() was `since`."""
    if model_error_count() <= since:
        return None
    _, text = max(e for e in (det.last_runtime_error, emb.last_runtime_error) if e)
    first_line = (text.strip().splitlines() or [text])[0].rstrip(".")
    msg = f"Model inference failed: {first_line}."
    if "no kernel image is available" in text:
        msg += (" Your GPU is not supported by this image. Use the :cuda-legacy image tag for "
                "older NVIDIA cards (GTX 9xx/10xx), or the :cpu tag.")
    return msg


_SCAN_WORKER = str(Path(__file__).parent / "scan_worker.py")


def run_scan(data_dir: str, *, manual: bool = False, scan_until: str | None = None,
             scan_since: str | None = None, migrate: bool = False,
             cancel: threading.Event | None = None, on_date=None, on_counts=None) -> tuple[dict, list]:
    """Run one poll cycle in an isolated subprocess, blocking until it finishes.

    Returns (counts, low_conf_assets). Raises RuntimeError on failure. If
    `cancel` is set while running, the subprocess is terminated."""
    args = {
        "data_dir": data_dir,
        "manual": manual,
        "scan_until": scan_until,
        "scan_since": scan_since,
        "migrate": migrate,
    }
    proc = subprocess.Popen(
        [sys.executable, _SCAN_WORKER],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1,
    )
    proc.stdin.write(json.dumps(args))
    proc.stdin.close()

    def watch_cancel():
        while proc.poll() is None:
            if cancel.wait(timeout=0.2):
                proc.terminate()
                return

    if cancel is not None:
        threading.Thread(target=watch_cancel, daemon=True).start()

    def relay_stderr():
        for line in proc.stderr:
            sys.stderr.write(line)

    stderr_thread = threading.Thread(target=relay_stderr, daemon=True)
    stderr_thread.start()

    result: dict = {}
    try:
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            t = msg.get("type")
            if t == "date" and on_date:
                on_date(msg["date"])
            elif t == "counts" and on_counts:
                on_counts(msg["counts"])
            elif t in ("done", "error"):
                result = msg
    except Exception:
        # Reading the protocol itself failed unexpectedly; don't leave the
        # worker running. On normal completion (EOF), just wait below instead:
        # the process may still be a few milliseconds into CUDA teardown.
        if proc.poll() is None:
            proc.terminate()
        raise
    finally:
        code = proc.wait()
        stderr_thread.join(timeout=5)

    if code != 0:
        raise RuntimeError(result.get("error") or f"scan worker exited with code {code}")
    return result.get("counts", {}), result.get("low_conf", [])
