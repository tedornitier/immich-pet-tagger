"""
Main entrypoint for immich-pet-tagger.
Starts the FastAPI enrollment UI and the background polling loop.
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from pathlib import Path
from embedder import load_embed_cache
from api import router as api_router
from device import pick_device
import inference
import detector as det
import embedder as emb

BASE_DIR = Path(__file__).resolve().parent
import state

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("main")

POLL_INTERVAL = int(os.environ.get("POLL_INTERVAL", 3600))
DATA_DIR = os.environ.get("DATA_DIR", "/data")
LONG_REQUEST_TIMEOUT = int(os.environ.get("LONG_REQUEST_TIMEOUT", 120))


async def polling_loop():
    device = pick_device()
    log.info(f"Poller started. Interval: {POLL_INTERVAL}s. Data dir: {DATA_DIR}. Device: {device}")
    migrated = False
    while True:
        try:
            log.info("Starting poll cycle...")
            async with state.scan_lock:
                state.scan_cancel.clear()
                await asyncio.to_thread(inference.run_scan, DATA_DIR, migrate=not migrated, cancel=state.scan_cancel)
                migrated = True
            log.info("Poll cycle complete.")
        except Exception as e:
            log.exception(f"Poll cycle failed: {e}")
        await asyncio.sleep(POLL_INTERVAL)


@asynccontextmanager
async def lifespan(app: FastAPI):
    state.init()
    load_embed_cache(Path(DATA_DIR))
    task = asyncio.create_task(polling_loop())
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


app = FastAPI(title="Immich Pet Tagger", lifespan=lifespan)

app.include_router(api_router)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/api/status")
async def status():
    return {
        "poll_interval": POLL_INTERVAL,
        "data_dir": DATA_DIR,
        "immich_url": os.environ.get("IMMICH_URL", "not set"),
        "yolo_ready": det.is_yolo_ready(),
        "clip_ready": emb.is_clip_ready(),
        "yolo_error": det.get_yolo_error(),
        "clip_error": emb.get_clip_error(),
    }


# FileResponse sets ETag/Last-Modified but no Cache-Control, so browsers fall back to
# heuristic caching and a plain reload can silently reuse a stale copy from before the
# last deploy without ever asking the server. no-cache forces revalidation on every
# load (still cheap: a 304 if the file hasn't changed) instead of relying on that guess.
NO_CACHE_HEADERS = {"Cache-Control": "no-cache"}


@app.get("/")
async def root():
    return FileResponse(str(BASE_DIR / "static" / "index.html"), headers=NO_CACHE_HEADERS)


@app.get("/accuracy.html")
async def accuracy_page():
    return FileResponse(str(BASE_DIR / "static" / "accuracy.html"), headers=NO_CACHE_HEADERS)


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
        timeout_keep_alive=LONG_REQUEST_TIMEOUT,
    )
