# immich-pet-tagger

Automatic pet tagging for Immich. Identifies your pets in new photos and tags them as people in Immich, the same way Immich tags human faces, but for cats, dogs, or any visually distinct subject.

Uses CLIP embeddings and a few reference photos you provide. No cloud services, no training required, runs entirely on your own hardware as a Docker sidecar alongside Immich.

![Pet Tagger UI showing a pet's possible missed photos and a past scan result](screenshot.png)

## How it works

1. You enroll your pets via a web UI: provide a few reference photos and a short description
2. A logistic regression classifier is trained locally on CLIP embeddings of those references
3. Every hour, new photos are scanned: YOLO detects and crops any animals in the photo, then each crop is embedded with CLIP and classified against your pets
4. Matching pets are tagged in Immich
5. Pets appear in Immich's People section just like humans

## Features

- **Import from Immich**: if Immich already recognizes your pet as a person, import them in one click. The tool picks up to 20 evenly distributed reference photos automatically.
- **Find similar photos**: uses a two-stage search to surface candidates. Your reference photos are used as visual queries against Immich's smart search, and the local classifier re-ranks the results by pet probability. Falls back to text search using your description when no refs exist yet.
- **Find candidates for "not my pets"**: samples random photos from your library, scores them with the classifier, and surfaces the top 60 most likely to confuse it for bulk review.
- **Negative samples**: mark photos that look like your pet but aren't, to sharpen the classifier's ability to reject false positives.
- **Date ranges**: restrict a pet to photos taken within a specific period (useful for pets that have passed away or were adopted later).
- **Scan controls**: set the scan start date and trigger a scan from the sidebar; the last scan stats are shown live.
- **Manage pets**: rename a pet, remove it from Pet Tagger only (keeps it and its tags in Immich untouched, so you can re-import later), delete it entirely (also removes the person and all its tags from Immich), or reset its Immich tags (untags every photo but keeps your curated reference photos so you can start tagging fresh).
- **Tagging accuracy** (📊 icon in the sidebar, or `/accuracy.html`): dry-run classifies a date range and compares it against your existing Immich tags, so you can see recall/false-positive rates and tune thresholds before trusting a full scan.

## Requirements

- Immich running and reachable over HTTP (tested with v2.7.5)
- Docker (on the same host or any machine that can reach Immich on the network)
- An Immich API key with the following permissions:

  | Permission | Reason |
  |---|---|
  | `asset.read` | Search results and asset metadata |
  | `asset.view` | Loading thumbnails |
  | `person.create` | Creating a new pet as a person in Immich |
  | `person.read` | Reading existing persons and thumbnails |
  | `person.update` | Renaming a pet |
  | `person.delete` | Deleting a pet |
  | `person.reassign` | Assigning a face to a person |
  | `face.create` | Writing face entries (the actual tagging) |
  | `face.read` | Checking existing faces on an asset |
  | `face.delete` | Removing face entries on ref removal or pet deletion |
  | `tag.create` | Only with `TAG_NAME`: creating the review tag on first use |
  | `tag.asset` | Only with `TAG_NAME`: applying the review tag to tagged photos |

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/tedornitier/immich-pet-tagger
cd immich-pet-tagger
```

### 2. Configure docker-compose.yml

Edit the following values:

```yaml
environment:
  - IMMICH_URL=http://immich-server:2283     # how this container reaches Immich
  - IMMICH_API_KEY=your_api_key_here         # generate one in Immich: Account Settings → API Keys
  - IMMICH_EXTERNAL_URL=http://localhost:2283 # how your browser reaches Immich (for photo links)
```

**Same Docker host as Immich:** use the container name as the hostname (e.g. `http://immich-server:2283`) and keep the shared network section at the bottom of the file. Find your Immich network name with `docker network ls`.

**Immich on a separate machine:** use its IP or hostname instead (e.g. `http://192.168.1.100:2283`) and change `external: true` to `external: false` at the bottom of `docker-compose.yml`.

### 4. Start the container

The default configuration runs on CPU, which works for any machine (including ARM64 like Raspberry Pi or Apple Silicon) without extra setup.

```bash
docker compose up -d
docker compose logs -f   # watch startup logs
```

If you want GPU acceleration, see [GPU support](#gpu-support) before running.

The first time the app actually needs to detect or classify a photo (a scan, or a UI action like Find suggestions), it downloads the YOLO model (~6 MB) and CLIP model (~350 MB) and saves them to the /data volume. After that, they load from disk and no internet connection is needed. The container must have internet access for this first download.


### Read-Only Root Filesystem

This image supports running with a read-only root filesystem. All writes (including model caches and configuration) are directed to the /data volume.

### 5. Open the UI

Go to **http://localhost:2287** in your browser.

The UI binds to `127.0.0.1` by default, so it is only reachable from the same machine. There is no authentication. To allow access from other devices on your network, or to use a different port, change the port binding in `docker-compose.yml`:

```yaml
ports:
  - "0.0.0.0:2287:8000"  # accessible from other devices on your network
```

To use a different port, change only the first number. The second number (`8000`) is the container's internal port and must stay as-is:

```yaml
ports:
  - "127.0.0.1:9000:8000"  # serves on port 9000 instead
```

Do not expose this to the internet without putting an authenticated reverse proxy in front of it.

## Updating

To update to a new version, pull the latest image and restart:

```bash
docker compose pull
docker compose up -d
```

This works for all variants since pre-built images are published for NVIDIA, AMD, and CPU-only.

---

## Getting started

Getting good results takes a few iterations. Start by adding a pet, building up references, and adding some negatives. Run a short test scan, review the results, refine, and repeat until you're satisfied. Then run the full backfill.

### Step 1: Add your pet

**Import from Immich**: use this if Immich already recognizes your pet as a person from its own face detection. This is ideal when the person in Immich contains only photos of that pet, for example if you tagged them manually and are confident the assignments are correct. The tagger does not remove or correct existing Immich face assignments, so any misidentified photos already tagged in Immich will stay tagged. If Immich's recognition was noisy, consider adding your pet manually instead.

1. Click **↓ Import from Immich** in the sidebar
2. Find and click your pet in the grid
3. Enter a short description (e.g. `orange tabby cat`) and an optional date range
4. Click **Import**. Up to 20 reference photos are imported automatically.

**Add manually**: use this if Immich doesn't know your pet yet.

1. Click **+ Add pet**, fill in the name, a short description (e.g. `black labrador dog`), and an optional date range
2. Click **Create**

The description is used by Immich's CLIP model to find the first batch of candidate photos. Keep it short: 2–4 descriptive keywords.

### Step 2: Add reference photos

References are what the classifier learns from. Quality matters more than quantity.

1. Select your pet in the sidebar and click **Find references**
2. Browse the results. They are ranked by visual similarity to your existing refs, or to your description if no refs exist yet.
3. Aim for 20–30 to start; results improve up to around 50. For each photo:
   - **Add to pet**: clear, close-up shot, your pet is the only subject.
   - **Ignore**: blurry, distant, another person or animal visible alongside your pet, or a look-alike that is not yours. Ignored photos won't appear again.
   - **Not a pet**: photos that could confuse the classifier. Empty rooms, other species, ambiguous shots. Around 50 is enough.

If you already know a specific photo you want to use, click **Add manually** and paste the Immich photo URL or asset ID directly.

### Step 3: Add "not a pet" samples

These teach the classifier what not to tag: empty rooms, other animals of a different species, ambiguous shots with no clear subject. Without them, the classifier will tag almost anything.

1. In the **Not a pet** panel (bottom right of the screen), click **Find candidates** to automatically surface more photos that might confuse the classifier
2. Select the relevant ones and click **Not a pet**
3. To add a specific photo directly, click **Add manually** and paste its Immich URL or asset ID


### Step 4: Run a test scan

Start with a recent date so the scan covers fewer photos, making it quicker to review and refine before committing to a full backfill.

1. In the **Scan from** panel at the bottom of the sidebar, set a date 1–2 weeks back
2. Click **Scan** and wait for the results
3. If **Review N low confidence** appears in the results, click it to see photos the classifier identified as a match but wasn't fully confident about.
4. Go through them: add correctly identified ones as references, and click **Ignore** on the rest. Ignored photos won't appear in future results.

### Step 5: Iterate

Repeat steps 2–4 a couple of times. Each round of added references and negatives improves accuracy. Results typically stabilize after 2–3 iterations.

If you want a more precise read on accuracy than eyeballing the scan results, use **Tagging accuracy** (📊 icon in the sidebar, or `/accuracy.html`): it dry-runs classification over a date range and compares it against your existing Immich tags, showing recall and false-positive rates broken down by pet, photo/video, and detection confidence, and lets you test different thresholds without waiting for a real scan.

### Step 6: Run the full backfill

Once you're happy with the accuracy on the test window:

1. Set the scan date to the earliest date you want to tag. A good starting point is the date you got your pet.
2. Click **Scan** to process all photos in that range

After that, the background poller runs every hour and tags new photos automatically. Your pets appear in Immich's **People** section.

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `IMMICH_URL` | `http://immich-server:2283` | Immich URL for container-to-container communication |
| `IMMICH_EXTERNAL_URL` | `http://localhost:2283` | Immich URL as seen from your browser, used for links |
| `IMMICH_API_KEY` | required | Immich API key |
| `TAG_NAME` | *(unset, `immich-pet-tagger` in docker-compose.yml)* | If set, this Immich tag is applied to a photo every time a face is written to it (background scan, manual scan, or enrolling a reference photo from the UI), giving you a review queue in Immich of everything the tagger touched. Change the value to whatever tag name you prefer, use `/` for a nested tag (e.g. `Pets/Review`), or remove the line to disable it. The tag is created on first use if it does not exist. Applied inline with each face, not batched, so a photo shows up under the tag as soon as it is tagged, and removed again if that face is later removed and no other pet's face remains on the photo. Removing the tag from a photo in Immich is safe — Pet Tagger never reads it back. Only applies going forward: enabling it does not retroactively tag photos faced before it was turned on. |
| `POLL_INTERVAL` | `3600` | Seconds between background scans. Models are loaded only during a scan and unloaded afterward to save RAM. |
| `CLIP_MODEL_NAME` | `ViT-B-16` | CLIP model architecture, passed to `open_clip.create_model_and_transforms`. See [open_clip's pretrained list](https://github.com/mlfoundations/open_clip/blob/main/docs/pretrained.md) for valid combinations with `CLIP_PRETRAINED`. Larger models need more RAM/VRAM per `GPU_WORKERS` thread. Embedding caches are namespaced per CLIP/YOLO model combo, so switching is safe and does not affect other models' cached data, but each new combo starts with a cold cache and re-embeds refs and assets on first use. |
| `CLIP_PRETRAINED` | `openai` | CLIP pretrained weights tag for `CLIP_MODEL_NAME`. |
| `YOLO_MODEL_NAME` | `yolov8n.pt` | YOLO detection model, passed to `ultralytics.YOLO`. Any Ultralytics-compatible weights file works (e.g. `yolov8s.pt`, `yolov8m.pt` for better accuracy at the cost of speed), as long as it's COCO-trained so the animal class IDs still line up. Larger models need more RAM/VRAM per `GPU_WORKERS` thread. |
| `SCAN_WORKERS` | `GPU_WORKERS × 32` | Concurrent thumbnail fetches. Auto-derived to keep GPU batches full. Override only if Immich feels slow during scans. |
| `GPU_WORKERS` | `2` (GPU) / `1` (CPU) | Parallel YOLO and CLIP inference threads. `2` is optimal for GPU; CPU defaults to `1` since a second worker just duplicates the models in RAM with no throughput gain. |
| `YOLO_INPUT_SIZE` | `640` | YOLO detection resolution in pixels. Higher values improve detection of small animals at the cost of more memory and compute. Must be a multiple of 32. |
| `YOLO_BATCH_SIZE` | `32` | Max images per YOLO inference batch. Reduce if you hit GPU out-of-memory errors. |
| `EMBED_CACHE_SIZE` | `5000` | Max number of embeddings kept in the in-memory LRU cache. Older entries are evicted when the limit is reached. |
| `THRESHOLD` | `0.8` | Min confidence (0–1) to tag a photo, when YOLO found a real crop. |
| `THRESHOLD_FALLBACK` | same as `THRESHOLD` | Min confidence (0–1) to tag a photo when YOLO found nothing and the whole image was embedded instead of a crop. Defaults to `THRESHOLD`'s value, so it does nothing unless set explicitly. Worth raising above `THRESHOLD`: the tagging accuracy tool has consistently measured this whole-image fallback as a noisier signal than a real crop, a separate, stricter threshold trades a bit of recall on fallback matches for meaningfully fewer false positives. |
| `WHOLE_IMAGE_MATCH` | `tag` | What a whole-image fallback match is allowed to do once it clears `THRESHOLD_FALLBACK`. `tag` (default) tags it like any other match — prior behaviour, unchanged. `review` never tags it and sends it to the low-confidence review queue instead, however high it scores, so the fallback's extra recall stays available but a human confirms each one; such photos are marked "Whole photo" in the review grid. `ignore` disables the fallback entirely: photos with no detection are skipped without even being embedded, which also speeds up scans on largely animal-free libraries. Use the tagging accuracy tool's `fallback` bucket to decide which fits your library. |
| `YOLO_CONF` | `0.25` (`0.2` in docker-compose.yml) | Min YOLO detection confidence (0–1) to count as a real crop. Lower catches more (small, turned-away, or partially visible pets) but crops get noisier. Below this, tagging falls back to embedding the whole photo instead of a crop. |
| `IOU_THRESHOLD` | `0.7` | YOLO's NMS IoU threshold (0–1). Overlapping detections whose boxes overlap more than this fraction are merged into one. Lower this if two cuddling/overlapping pets in the same photo are being collapsed into a single detection. |
| `LONG_REQUEST_TIMEOUT` | `120` | Max seconds for CPU-heavy UI requests (Find missed, Find candidates, Find references). Responses stream keepalive bytes so browsers do not drop idle connections. |

---

## GPU support

The default setup runs on CPU and requires no extra configuration. A GPU makes scans significantly faster but requires additional setup. Pre-built images are published for CPU (AMD64 and ARM64), NVIDIA (AMD64), and AMD/ROCm (AMD64).

**CPU (default):** no changes needed.

**NVIDIA GPU:** install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) on your host, then in `docker-compose.yml`:
1. Change the image tag to `:latest` (default) or `:cuda-legacy` (Maxwell/Pascal/Volta — see below)
2. Uncomment the `deploy:` section
3. Set `GPU_WORKERS=2`

```yaml
image: ghcr.io/tedornitier/immich-pet-tagger:latest
```

The `:latest` image ships PyTorch's CUDA 12.8 wheels and supports Turing, Ampere, Ada Lovelace, Hopper, and Blackwell GPUs (compute capability 7.5–12.0, e.g. RTX 20xx/30xx/40xx/50xx, Tesla T4/A100/H100). Pascal and older are not covered — see the legacy tag.

**NVIDIA legacy GPU (`:cuda-legacy`):** for Maxwell, Pascal, and Volta cards (GTX 9xx/10xx, Tesla P100/V100, compute capability 5.0–7.0). Same setup as above but use the `:cuda-legacy` tag instead:

```yaml
image: ghcr.io/tedornitier/immich-pet-tagger:cuda-legacy
```

This variant uses PyTorch's CUDA 12.6 wheels, which still include kernels for `sm_50` through `sm_90` but drop Blackwell (`sm_100`/`sm_120`). If you see `CUDA error: no kernel image is available for execution on the device` with `:latest` on an NVIDIA card, switch to this tag.

**AMD GPU:** install ROCm drivers, then in `docker-compose.yml`:
1. Change the image tag to `:rocm`
2. Uncomment the `deploy:` section and change the driver to `amdgpu`

```yaml
image: ghcr.io/tedornitier/immich-pet-tagger:rocm
```
```yaml
driver: amdgpu
```

CPU-only works fine for most home libraries. Expect roughly 10x slower processing compared to GPU.

---

## Bigger models (optional)

The default models (`YOLO_MODEL_NAME=yolov8n.pt`, `CLIP_MODEL_NAME=ViT-B-16`/`CLIP_PRETRAINED=openai`) are chosen to run comfortably on CPU, including low-power hardware like a Raspberry Pi. If you have GPU headroom and want to trade resources for accuracy, `YOLO_MODEL_NAME=yolov8s.pt` with `CLIP_MODEL_NAME=ViT-L-14`/`CLIP_PRETRAINED=openai` measured ~40% fewer missed tags and ~13% fewer false positives than the defaults on a real library (see `.claude/decisions.md`). This is opt-in, not the shipped default, for two reasons:

- `ViT-L-14` is roughly 4x the parameters/compute of the default `ViT-B-16`, and needs a larger download (~900MB vs ~350MB) plus more RAM/VRAM per `GPU_WORKERS` thread. That conflicts with the project's CPU-friendly default.
- The accuracy gain was measured on one library with several hundred reference photos per pet. A classifier with very few refs may not benefit as much from the larger embedding space, since there's less data to fit a good decision boundary with.

In my own personal testing I reached 96% tagging accuracy at a 1.4% false-positive rate using this combo (`yolov8s.pt` + `ViT-L-14`/`openai`), versus the defaults.

If you want to try it, set in `docker-compose.yml`:

```yaml
- YOLO_MODEL_NAME=yolov8s.pt
- CLIP_MODEL_NAME=ViT-L-14
- CLIP_PRETRAINED=openai
```

Embedding caches are namespaced per model combo, so switching is safe and reversible, it just means a cold cache and re-embedding refs/assets on first use. Use **Tagging accuracy** (📊 icon in the sidebar, or `/accuracy.html`) to compare recall/false-positive rates against your current setup before committing to the switch on a full scan.

## Limitations

- **YOLO fallback**: when no animals are detected by YOLO, the full image is classified as a whole and only one pet can be tagged per photo
- **Polling only**: photos are processed on the next poll cycle (default every hour), not instantly

## Troubleshooting

**Pet not appearing in Immich after enrollment**
Immich only shows people with at least one face assigned. Add at least one reference photo and wait for a poll cycle.

**Low accuracy / wrong pet tagged**
Add more reference photos, add more negative samples, or lower the threshold in `docker-compose.yml`.

**Container can't reach Immich**
Make sure the network name in `docker-compose.yml` matches the output of `docker network ls`.

**Thumbnail proxy returns 401**
Your API key is missing `asset.view` permission.
