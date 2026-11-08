"""
Main entrypoint for immich-pet-tagger.
Starts the FastAPI enrollment UI and the background polling loop.
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from urllib.parse import urlsplit

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

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
# Extra origins allowed to send state-changing requests, e.g. "https://pets.example.com".
# Only needed behind a reverse proxy that rewrites the Host header, for browsers that do
# not send Sec-Fetch-Site. Comma separated.
TRUSTED_ORIGINS = {o.strip().rstrip("/").lower() for o in os.environ.get("TRUSTED_ORIGINS", "").split(",") if o.strip()}
SAFE_METHODS = {"GET"}


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


def is_cross_site(method: str, headers) -> bool:
    """True if a browser sent this state-changing request on behalf of another website.

    Cross-site form posts and no-cors fetches skip the CORS preflight, and FastAPI parses a
    body without Content-Type as JSON, so without this check any page open in the browser
    could start scans, change pets or reset them in Immich. Requests without Origin (curl,
    scripts) are not browser cross-site requests and pass."""
    if method in SAFE_METHODS:
        return False
    site = headers.get("sec-fetch-site")
    if site == "cross-site":
        return True
    # Set by the browser itself, a page cannot forge it. Checked before Origin because some
    # referrer policies make browsers send "Origin: null" for same-origin form posts, and a
    # reverse proxy that rewrites Host would break the Origin comparison below.
    if site in ("same-origin", "none"):
        return False
    origin = headers.get("origin")
    if origin is None:
        return False
    if origin == "null":
        return True
    origin = origin.rstrip("/").lower()
    if origin in TRUSTED_ORIGINS:
        return False
    # Only host and port are compared, not the scheme: behind a TLS-terminating proxy the
    # browser is on https while the app itself only sees http.
    try:
        netloc = urlsplit(origin).netloc
    except ValueError:  # malformed Origin, e.g. "http://[::1"
        return True
    return netloc != headers.get("host", "").lower()


app = FastAPI(title="Immich Pet Tagger", lifespan=lifespan)


@app.middleware("http")
async def block_cross_site_requests(request, call_next):
    if is_cross_site(request.method, request.headers):
        log.warning(f"Blocked cross-site {request.method} {request.url.path} (origin: {request.headers.get('origin')})")
        return JSONResponse({"detail": "Cross-site request blocked."}, status_code=403)
    return await call_next(request)


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
