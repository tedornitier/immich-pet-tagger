# Changelog

## Unreleased

### Features
- `WHOLE_IMAGE_MATCH` env var decides what a whole-image fallback match (YOLO found no animal to crop) may do once it clears `THRESHOLD_FALLBACK`. Where `THRESHOLD_FALLBACK` tunes *how confident* such a match must be, this decides what happens when it is: `tag` (default) tags it like any other match, exactly as before; `review` never tags it and routes it to the low-confidence review queue instead, however high it scores, keeping the fallback's extra recall while putting a human in the loop; `ignore` disables the fallback entirely, skipping such photos before they are even embedded, which also cuts scan cost on largely animal-free libraries. Whole-image entries in the review queue are marked "Whole photo", since accepting one as a reference teaches the classifier to match on background rather than on the pet. Scan stats gain a "Whole photo" count, shown only when the setting is not `tag`.

## v1.7.5

### Fixed
- "Untag all photos in Immich" (the reset button in a pet's delete menu) deleted the Immich person but never removed the review tag (`TAG_NAME`) from the affected photos, the same gap the v1.7.4 fix closed for the ref-removal and whole-pet-delete paths but missed here (part of #46).

## v1.7.4

### Fixed
- Removing a pet's face (single ref or the whole pet) could leave the review tag (`TAG_NAME`) on the photo instead of stripping it. The untag step read a cached tag id directly instead of resolving it on demand like the tag-apply step does, so it silently did nothing whenever it was the first review-tag call made by a freshly started container.
- The tagging accuracy tool's recall/false-positive curve chart could render past the 100% mark due to axis padding, and didn't indicate the chart is clickable to set a threshold.

## v1.7.3

### Features
- `THRESHOLD_FALLBACK` env var sets a separate confidence threshold for whole-image fallback classifications (when YOLO finds no animal to crop), instead of sharing `THRESHOLD` with real crops. Defaults to `THRESHOLD`'s value, so it's a no-op unless set explicitly. The tagging accuracy tool has consistently measured the fallback path as a meaningfully noisier signal than a real crop; this lets that be corrected for in live tagging (e.g. setting it higher than `THRESHOLD`) instead of only being visible in the diagnostic.

## v1.7.2

### Features
- `IOU_THRESHOLD` env var exposes YOLO's NMS overlap-suppression threshold (default `0.7`, matching prior behavior). Lowering it helps when two pets photographed close together or cuddling get merged into a single detection instead of two separate crops (addresses #44).

### Fixed
- Pet names containing quotes or apostrophes (e.g. `D'argo`) broke the sidebar's edit/delete buttons, since the name was interpolated directly into an inline `onclick` string. Names are now passed through a `data-name` attribute instead (fix contributed by @thespragg in #45). The low-confidence scan panel's pet-assignment buttons had the same unescaped-name issue and are fixed too.

## v1.7.1

### Fixed
- The background scanner checked a pet's date range against the photo's Immich upload time instead of the date it was actually taken, so a photo imported/synced after a pet's `since` date could pass the range check even though it was taken years earlier (part of #43).

## v1.7.0

### Features
- **Review tag**: setting `TAG_NAME` makes the tagger apply that Immich tag to a photo every time it writes a face to it, giving you a single place in Immich to review everything it touched. The tag is created on first use, and applied inline with each face rather than batched, so a photo appears under the tag as soon as it is tagged. The tag is also removed again if a pet's face is removed from a photo, whether by removing a single ref or deleting the pet entirely, as long as no other pet's face remains on it, so the review queue doesn't accumulate photos the tool no longer has anything to say about. Only applies going forward: enabling it does not retroactively tag photos faced before it was turned on. Requires `tag.create` and `tag.asset` on the API key.
- **Tagging accuracy** (📊 icon in the sidebar, or `/accuracy.html`): dry-run classifies every photo and video in a date range and compares against your actual Immich tags, so you can measure real recall and false-positive rates instead of guessing. Breaks results down by pet, photo/video, and whether YOLO found a detection or fell back to the whole image, with a summary table plus thumbnail grids of only the mismatches. A threshold explorer sweeps every CLIP threshold from 0.30 to 0.99 using the scores already collected by the run (no re-scan) and plots recall vs. false-positive rate per bucket, click a chart to set that bucket's threshold. YOLO detection confidence can be overridden per run to test a different cutoff (thumbnails and raw detections are cached per asset, so re-running with a new value is fast); CLIP thresholds are picked entirely client-side and never trigger a re-run. Nothing is saved to live settings, only that one run. Results can be downloaded and re-imported later to skip re-running. Not part of the tagging pipeline, a diagnostic tool for tuning thresholds. Production `THRESHOLD`/`YOLO_CONF` are still configured via `docker-compose.yml` only.
- **Low-confidence review filtered by predicted pet**: with several pets enrolled, low-confidence results used to show as one mixed grid. Filter chips now group them by which pet the classifier guessed, so reassigning is faster (addresses #42).

### Changed
- `docker-compose.yml` now ships `YOLO_CONF=0.2` explicitly (was previously unset, falling back to the code default of `0.25`). Measured to catch more real detections without a meaningful false-positive cost using the tagging accuracy tool.
- README documents a "Bigger models" opt-in upgrade (`YOLO_MODEL_NAME=yolov8s.pt`, `CLIP_MODEL_NAME=ViT-L-14`, `CLIP_PRETRAINED=openai`) for users with GPU headroom, measured ~40% fewer missed tags and ~13% fewer false positives than the defaults on a real library (96% tagging accuracy at a 1.4% false-positive rate in my own testing). Not the shipped default, since it needs more RAM/VRAM and a larger model download.

### Fixed
- Low-confidence review could suggest a pet for a photo outside that pet's configured date range (e.g. a guess for a pet who wasn't even adopted yet). The date-range check now runs before the confidence check, so an out-of-range guess is dropped outright instead of surfacing in the review queue.

## v1.6.1

### Fixed
- Low-confidence review thumbnails now show the actual cropped animal instead of the whole photo, instead of discarding the detected bounding box before it reached the review queue (fix contributed by @vegardengen in #36).
- Tagging a pet from the UI (including the low-confidence review grid) now creates the Immich face box over the detected animal region instead of covering the entire image.

## v1.6.0

### Features
- **Scans now run in an isolated subprocess**: each background or manual scan loads YOLO/CLIP in a short-lived child process that exits when the scan finishes, so the OS fully reclaims its RAM/VRAM instead of it lingering in the main process. Default background poll interval is now 1 hour (`POLL_INTERVAL=3600`). UI-only actions (suggestions, borderline, negatives, import) still load models on demand in-process and unload them when idle.

### Changed
- YOLO and CLIP no longer start loading at container boot. Models now load on first use (a scan or a UI inference action) instead, and the "models not ready" banner only appears if loading actually fails, not while a model is loading.

### Fixed
- Host RAM freed by unloading YOLO/CLIP after a UI inference action (suggestions, borderline, negatives, import) now actually returns to the OS. glibc gives each thread its own malloc arena and only trims the main one, so memory freed by the worker threads was staying mapped even after `gc.collect()`; the container now runs with `MALLOC_ARENA_MAX=1` plus an explicit `malloc_trim(0)` after unload.
- Scans now actually skip loading YOLO/CLIP when there's nothing new to process. The scan subprocess wasn't loading the on-disk embedding cache, so it re-embedded every pet's reference and negative photos with CLIP on every single scan, even ones that found zero new assets. Separately, ref photos that store a crop bounding box were never being written to that cache at all, so they'd keep missing forever regardless. Both are fixed; a scan with fully cached refs and no new assets now completes without touching YOLO or CLIP.
- Clicking "Scan Now" while the hourly background scan is running now actually cancels it and starts the manual scan immediately, instead of silently queuing behind it until the background scan finishes.
- The scan subprocess could crash on exit with `terminate called without an active exception` (SIGABRT) right after completing a scan successfully, due to a PyTorch/CUDA teardown race during normal Python shutdown. It now exits via `os._exit()` instead, skipping that teardown entirely.
- The background poller could re-fetch and re-process the same already-tagged photo every single cycle forever, forcing a full YOLO/CLIP load each time even with nothing new to do. The saved poll cursor was the last asset's own timestamp, and Immich's `createdAfter` filter is inclusive; the cursor now advances 1ms past the latest asset before being saved.

## v1.5.2

### Features
- **Selectable CLIP and YOLO models**: `CLIP_MODEL_NAME`/`CLIP_PRETRAINED` and `YOLO_MODEL_NAME` env vars let you trade accuracy for speed/memory (e.g. a larger CLIP backbone, or a bigger YOLO model for better detection). Embedding caches are namespaced per model combo so switching never mixes stale data from a previous model.
- Manual scan's "Until" date now defaults to today, replacing the old "From" default (the poller's last-run timestamp), matching the common "scan the last N days" workflow.

### Fixes
- **Manual scan no longer corrupts the background poller's progress**: manual scans and the automatic poller used to share one cursor timestamp file but interpreted it differently (EXIF taken-date vs upload time). Resetting the date for a manual scan could silently rewind the poller's cursor too, causing it to pick up far more assets than expected on bulk-imported libraries. Manual scan now uses its own date range and never touches the poller's cursor.

## v1.5.1

### Fixes
- **Model download failures no longer hang silently**: YOLO and CLIP workers now start at container boot. If a download fails (e.g. no internet on first start), the error is logged immediately and any inference action returns a clear error instead of hanging forever.
- **UI warning banner**: a banner is shown when models are not ready yet, with instructions to ensure internet access on first start.
- Immich 3.0.0 search endpoints now correctly use the `size` parameter.

## v1.5.0

### Features
- **Faster "Find references"**: the trained classifier is now cached and crop embeddings are stored per asset in a local SQLite cache (`crops.db`), so repeat runs reuse prior work and return almost instantly instead of rescoring every candidate. The cache is keyed by asset and grows only with photos actually processed, not the whole library.
- **Add references and negatives by link**: new "Add manually" button in both the references panel and the "Not a pet" panel. Paste any Immich photo URL or bare asset ID to add it directly, without going through the suggestion flow.
- **ARM64 image**: the `:cpu` image is now published for `linux/arm64` (Raspberry Pi, Apple Silicon) in addition to `amd64`.

### Fixes
- **Stable confidence scores**: negative sample selection during classifier training is now deterministic, so identical references and negatives always produce the same percentages instead of shifting between runs.
- **Read-only root filesystem**: YOLO and CLIP model downloads are now redirected to the `/data` volume. Mount only `/data` for hardened deployments.
- Long UI requests (Find missed, Find candidates, Find references) now stream keepalive bytes during CPU-heavy scoring so browsers no longer drop idle connections. Server-side limit is 120s (`LONG_REQUEST_TIMEOUT`).

---

## v1.4.0

### Features
- **Untag all photos in Immich**: new option in the pet delete modal that removes all Immich face tags and creates a fresh person, while keeping local reference images so you can start tagging again immediately.

### Fixes
- **Partner sharing**: assets not owned by the API key holder are now skipped during tagging. Previously, when partner sharing was enabled, the tagger would create cross-account face records that caused a `FOREIGN KEY constraint failed` crash on the partner's mobile sync stream.
- Immich API errors are now propagated instead of silently returning empty results when asset searches fail.
- All data file writes are now atomic (write to tmp, then replace) to prevent file corruption on crash or power loss.
- All JSON reads now guard against corrupted or partial files, returning safe defaults instead of crashing.
- Embed cache is now capped with LRU eviction and batched disk saves, preventing unbounded memory growth on large libraries.
- Fetch thumbnail timeout increased to 30s, fixing failures on remote or slower Immich setups.
- Network errors (Immich unreachable, timeout) now surface as readable messages in the UI instead of a generic 500 error.

---

## v1.3.3

### Features
- Scan panel now accepts an optional end date, limiting the scan to a specific date range.
- New Stop button appears during a scan and cancels it immediately.

### Fixes
- Photos marked as "Not a pet" are now excluded from the low confidence review panel.
- Renamed "Not my pets" to "Not a pet" to avoid the misleading implication that it means "another pet".
- Manual scan start date is now capped at the current time, preventing a future date from stalling the poller after the scan completes.

---

## v1.3.2

### Fixes
- Manual scans now use EXIF date for the date picker, matching what users see in the Immich library. The background poller continues using upload time to avoid skipping late-synced photos.

---

## v1.3.1

### Fixes
- CPU image reduced from ~12.7 GB to ~2.4 GB. The default build was silently pulling the CUDA torch wheel from PyPI instead of the CPU-only wheel.
- Switched to a multi-stage Docker build with a virtual environment, eliminating ghost install layers that kept deleted packages on disk.
- Removed triton (545 MB, only needed for `torch.compile()`) from all images.
- Fixed a duplicate opencv library left behind by the previous `--force-reinstall` approach.

---

## v1.3.0

### Crop-centric references
- References now store the YOLO-detected crop rather than the full image, matching the format seen during inference. Existing refs are migrated automatically on startup.
- The ref grid now shows the crop thumbnail instead of the full photo.
- Import from Immich and Find references now run YOLO on candidates and skip photos where no animal is detected.
- When YOLO finds no crop for a ref, the full image is used as a fallback instead of silently dropping the ref.

### UX improvements
- Updated getting started guide with guidance based on tested results: aim for 20–30 references, ~50 negatives. Action buttons now have matching tooltips. Skip renamed to Ignore.
- Refs panel no longer scrolls to the top when removing a ref.
- Fixed an intermittent bug where clicking a pet in the sidebar would show the getting started guide instead of the pet view.
- Find candidates now samples 50 photos instead of 100 (faster) and raises the minimum score threshold to 30%.

### Other
- Scan progress now uses upload time (`createdAt`) instead of EXIF date, so late-synced photos never fall behind the scan cutoff.
- Ref and negative grid thumbnails now use `object-fit: contain`.
- Default UI port changed to 2287.

---

## v1.2.0

### Features
- **Version indicator**: the header now shows the current version and notifies you when a new release is available.
- **Persistent model cache**: the YOLO and CLIP models are cached in a named Docker volume, so they survive container updates without re-downloading.

### Fixes
- **CPU-only default**: `docker-compose.yml` now defaults to the `:cpu` image with GPU acceleration as an opt-in. Uncomment the `deploy` block and switch to the `:latest` image to enable NVIDIA GPU.
- **UI bound to localhost**: the UI now binds to `127.0.0.1` by default, preventing unauthenticated access from other devices on the network.
- **Low confidence count**: the "Review N low confidence" count in scan results now matches the number of photos actually shown in the review panel.
- **Not my pets count**: the label now always shows a plain number instead of extra status text.
- **UI font alignment**: minor font consistency fix.

---

## v1.1.0

### Fixes
- **Borderline threshold**: the low-confidence review threshold was hardcoded to 0.85 instead of tracking the configured `THRESHOLD`. Now derived dynamically as `[THRESHOLD * 0.375, THRESHOLD]`.
- **Import from Immich: missing face IDs**: refs imported via "Import from Immich" had `face_id: null`, preventing clean deletion later. Face IDs are now extracted during import.

### Docs
- Pre-built images published to GHCR for NVIDIA (`:latest`), AMD/ROCm (`:rocm`), and CPU-only (`:cpu`). No build step needed.
- Clarified that the tagger can run on a separate machine from Immich. Set `IMMICH_URL` to Immich's IP or hostname and change `external: true` to `external: false` at the bottom of `docker-compose.yml`.
- Added update instructions: `docker compose pull && docker compose up -d`.

---

## v1.0.0

### Performance
- **Batch GPU inference**: CLIP workers dequeue requests from all scan threads and process them as batches, keeping the GPU fully utilised. Scans are significantly faster on large libraries.
- **Concurrent thumbnail fetching**: assets are fetched in parallel with a configurable thread pool (`SCAN_WORKERS`, auto-derived as `GPU_WORKERS × 32`).
- **YOLO runs on CUDA**: the YOLO detector now loads onto the GPU when available.

### Features
- **AMD/ROCm support**: Docker image supports NVIDIA (default), AMD/ROCm, and CPU-only via a build arg.
- **In-UI getting started guide**: the main panel shows a 6-step workflow on first open. The `i` button in the sidebar header brings it back at any time.
- **Pet folder keys use person ID**: pet data folders are keyed by Immich person UUID instead of name, avoiding issues with special characters.

---

## v0.4.0

### Features
- **YOLO animal detection**: bounding boxes are computed before classification. Multi-pet photos are handled per crop, so each animal in the frame is classified separately. Improves accuracy and enables tagging photos with more than one pet.
- **Visual ref search**: "Find similar photos" now uses ref asset images as the CLIP query instead of text description, producing far more relevant candidates.
- **Find missed photos**: new "Find missed" button scores borderline candidates with the classifier and surfaces photos just below the confidence threshold. Useful for finding good refs to improve recall.
- **Score-based negatives calibration**: "Find not my pets" uses the classifier to exclude photos the model already considers pets, ensuring only genuinely ambiguous photos are surfaced as negatives.
- **Live scan**: the scan panel shows live per-category counts (Tagged, Low conf., Other, Already tagged) and the current photo date while scanning. Triggering a new scan cancels any in-progress one.
- **Review low confidence**: after a scan, a new "Review X low confidence" button lists photos the classifier identified as a pet but scored below threshold, sorted by score with color-coded badges.
- **Open in Immich**: each thumbnail in all photo grids now has a direct link icon to open the asset in Immich.
- **Embedding cache persisted to disk**: CLIP embeddings are saved to `data/embeddings.pkl` and reloaded on startup, so restarts no longer re-embed all ref and negative photos.

### Fixes
- **Critical: duplicate face prevention**: `"person": null` faces (Immich-detected but unassigned) caused the existing-faces check to throw and return an empty set, bypassing deduplication and creating duplicate tags. Fixed with a null guard in all face lookup paths.
- **Pagination cap removed**: asset search was silently stopping at 1000 results due to a `total` field that Immich caps at 1000. Now paginates correctly until the page is smaller than the page size.
- **Scan dedup across YOLO crops**: a person was not marked as tagged when face_id retrieval failed after a successful POST, causing the same pet to be re-tagged on subsequent crops of the same photo.
- **Pet delete**: per-ref face deletion loop removed. Deleting the Immich person cascades face removal automatically.
- **activePet stale reference**: "Find missed" button stayed disabled after re-enrollment because the in-memory pet reference was not refreshed after `loadPets()`.
- Responsive photo grid (auto-fill columns instead of fixed 4).
- Browser locale used for date formatting in tooltips and date inputs.

### Docs
- README rewritten with features overview, docker-compose setup, and enrollment tutorial.
- Added guidance on picking good reference photos (skip multi-pet, blurry, or ambiguous frames).

---

## v0.3.0

### Features
- **Import from Immich**: import a pet directly from an existing Immich person. The tool fetches up to 20 evenly distributed single-pet photos as refs automatically, skipping photos where multiple named people appear.
- **Find candidates for "not my pets"**: new button searches across all pets simultaneously, merges results, scores them by pet-likeness using the classifier, and shows the top 60 in the main grid for bulk review.
- **Tool-only delete**: when deleting a pet, a third option lets you remove it from the tool only (keeping the Immich person and all tagged photos intact). Assets are never deleted in either case.
- **Clear all refs / Clear all negatives**: bulk-clear buttons with confirmation, local only, no Immich changes.

### Fixes
- "Find similar photos" now skips the CLIP classifier stage when the pet has no refs, returning text search results immediately instead of waiting for model inference.
- Auto-select newly created pet after adding it.
- Stay on the edited pet after saving edits (was jumping to the first pet).
- Import no longer crashes on photos where Immich returns a null person in the faces list.

### Internal
- `app/` directory volume-mounted in docker-compose: Python, HTML, CSS, and JS changes apply after `docker compose restart` with no rebuild needed.

---

## v0.2.0

### Features
- **Similar photo suggestions**: new "Find similar photos" button runs a two-stage pipeline — Immich smart search pre-filters by pet description, then the classifier ranks candidates by pet class probability. Replaces the manual search bar.
- **Pet description field**: each pet now has a short description (2-4 keywords, e.g. "orange tabby cat") used as the CLIP text query for suggestions.
- **Shift+click range selection**: hold Shift and click a second photo to select the full range in the grid.
- **Tagged photos panel**: view photos already tagged for the active pet; remove a tag or mark as "not my pets" in bulk.
- **Scan status panel**: shows last scan time, badge (running/idle/error/never), and per-pet stats (tagged, skipped, errors) after each poll cycle.
### UI
- Refs and "Not my pets" grids: 3-column layout, equal height, independently scrollable.
- "Not my pets" panel (formerly Negatives): unified card style with refs, clickable thumbnails linking to Immich.
- "Find similar photos" button moved to the top of the right panel as the primary action.
- Scan and last scan controls moved to the bottom of the sidebar.
- Removed the manual search bar.

### Internal
- Negative samples subsampled to 3x pet refs in the classifier to keep class balance without discouraging large negative sets.
- Static files split into `style.css` and `app.js` (was a single `index.html`).
- Backend refactored: `data.py` for file I/O, `immich.py` for HTTP helpers (replaces `immich_apis.py`).

---

## v0.1.0

Initial release. Core tagging loop, pet enrollment UI, ref/negative management, logistic regression classifier on CLIP embeddings.
