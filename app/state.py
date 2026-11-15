import asyncio
import threading

scan_lock: asyncio.Lock | None = None
scan_cancel: threading.Event = threading.Event()
scan_generation: int = 0
neg_progress: dict = {"current": 0, "total": 0, "running": False}
neg_request_id: int = 0
borderline_progress: dict = {"current": 0, "total": 0, "running": False}
borderline_request_id: int = 0
manual_scan_result: dict | None = None
scan_low_conf_assets: list = []
benchmark_progress: dict = {"current": 0, "total": 0, "running": False}
benchmark_generation: int = 0
benchmark_cancel: threading.Event = threading.Event()
benchmark_result: dict | None = None
refanalysis_progress: dict = {"current": 0, "total": 0, "running": False}
refanalysis_generation: int = 0
refanalysis_cancel: threading.Event = threading.Event()
refanalysis_result: dict | None = None
neganalysis_progress: dict = {"current": 0, "total": 0, "running": False}
neganalysis_generation: int = 0
neganalysis_cancel: threading.Event = threading.Event()
neganalysis_result: dict | None = None

# Trained classifier cache shared across API endpoints. Rebuilt only when the
# set of refs or negatives changes (tracked via fingerprint). Reusing the same
# trained model also keeps prediction scores stable between requests.
classifier_cache: dict | None = None
classifier_cache_lock = threading.Lock()


def init():
    global scan_lock
    scan_lock = asyncio.Lock()
