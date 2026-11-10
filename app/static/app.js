let pets = [], activePet = null, activeList = null, listCounts = { neg: 0, ignored: 0 }, selectedCrops = new Map(), refsItems = [], negEntries = [], immichUrl = 'http://localhost:2283', resultsMode = null, resultsGen = 0, lastClickedKey = null, progressTimer = null, scanLowConfAssets = [], scanLowConfThreshold = 0.8, scanLowConfFilter = null;

async function api(path, opts = {}) {
  const r = await fetch(path, { headers: { 'Content-Type': 'application/json' }, ...opts, body: opts.body ? JSON.stringify(opts.body) : undefined });
  if (!r.ok) {
    let t = await r.text().catch(() => r.statusText);
    // FastAPI errors are {"detail": "..."}, show the message instead of the raw JSON.
    try { const j = JSON.parse(t); if (j && typeof j.detail === 'string') t = j.detail; } catch(_) {}
    throw new Error(t);
  }
  const d = await r.json();
  // Streamed endpoints have already sent a 200 by the time they fail, the error is in the body.
  if (d && d.stream_error) throw new Error(d.stream_error);
  return d;
}

function toast(msg, type = '') {
  const el = document.getElementById('toast');
  el.textContent = msg; el.className = 'toast show' + (type ? ' ' + type : '');
  clearTimeout(el._t); el._t = setTimeout(() => el.className = 'toast', 4000);
}

function initials(name) { return name.slice(0, 2).toUpperCase(); }

const TILE_SIZE_KEY = 'pet_tagger_tile_size', TILE_SIZE_MIN = 80, TILE_SIZE_MAX = 320;

function changeTileSize(delta) {
  const current = parseInt(localStorage.getItem(TILE_SIZE_KEY), 10) || 120;
  const next = Math.min(TILE_SIZE_MAX, Math.max(TILE_SIZE_MIN, current + delta));
  document.documentElement.style.setProperty('--tile-size', next + 'px');
  localStorage.setItem(TILE_SIZE_KEY, next);
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

async function refreshState() {
  try {
    const cfg = await api('/api/config');
    immichUrl = cfg.immich_external_url.replace(/\/$/, '');
    const banner = document.getElementById('modelsBanner');
    if (cfg.models_error) {
      banner.innerHTML = '<strong>Model load failed.</strong> ' + cfg.models_error + ' On first use, yolov8n.pt (~6 MB) and the CLIP model (~350 MB) are downloaded. Ensure the container has internet access, then retry. To use offline, copy the model files to the data volume (see README).';
      banner.classList.add('visible');
    } else {
      banner.classList.remove('visible');
    }
  } catch(e) {}
  await loadPets();
  loadNegatives();
  loadIgnored();
}

// ---------------------------------------------------------------------------
// Pets
// ---------------------------------------------------------------------------

async function loadPets() {
  try {
    const d = await api('/api/pets');
    pets = d.pets;
    if (activePet) setSelection(pets.find(p => p.name === activePet.name) || activePet);
    renderSidebar();
  } catch(e) { toast('Could not load pets: ' + e.message, 'error'); }
}

function renderSidebar() {
  const el = document.getElementById('petsList');
  renderListRows();
  if (!pets.length) {
    el.innerHTML = '<div style="padding:16px;font-size:12px;color:var(--text3);text-align:center;line-height:1.6;">No pets yet.<br>Add one to get started.</div>';
    if (!activeList) {
      showGuide();
    }
    return;
  }
  el.innerHTML = pets.map(p => `
    <div class="pet-item ${activePet?.name === p.name ? 'active' : ''}" data-name="${escapeHtml(p.name)}" onclick="selectPet(this.dataset.name)">
      <div class="pet-avatar">${p.person_id ? `<img src="/api/person-thumb/${p.person_id}" onerror="this.parentElement.textContent=initials(this.parentElement.parentElement.dataset.name)" alt="">` : initials(p.name)}</div>
      <div class="pet-info">
        <div class="pet-name">${escapeHtml(p.name)}</div>
        <div class="pet-count">${p.ref_count} ref${p.ref_count !== 1 ? 's' : ''}</div>
      </div>
      <button class="pet-edit" onclick="event.stopPropagation(); openEditPet(this.closest('.pet-item').dataset.name)" title="Edit">✎</button>
      <button class="pet-delete" onclick="event.stopPropagation(); openDeletePet(this.closest('.pet-item').dataset.name)" title="Delete">✕</button>
    </div>`).join('');
}

// "Not a pet" and "Ignored" rows under the pets, styled like pet rows.
function renderListRows() {
  document.getElementById('listRows').innerHTML = Object.entries(TILE_LISTS).map(([kind, L]) => `
    <div class="pet-item ${activeList === kind ? 'active' : ''}" onclick="selectList('${kind}')">
      <div class="pet-avatar list-avatar">${L.glyph}</div>
      <div class="pet-info">
        <div class="pet-name">${L.title}</div>
        <div class="pet-count">${listCounts[kind]} photo${listCounts[kind] !== 1 ? 's' : ''}</div>
      </div>
    </div>`).join('');
}

// The sidebar selection: a pet (an object from pets), a tile list ('neg' | 'ignored'), or
// nothing (the guide). Every change goes through here so the two can't both be set.
function setSelection(pet, list = null) {
  activePet = pet || null;
  activeList = pet ? null : list;
  if (!pet) refsItems = [];
}

// The "i" button: leave the current pet or list, then show the guide on its own.
function openGuide() {
  if (!leaveSelection()) return;
  setSelection(null);
  renderSidebar();
  showGuide();
}

function showGuide() {
  resetResults();
  setCenterView('guide');
  document.getElementById('centerBar').style.display = 'none';
  document.getElementById('photoGrid').innerHTML = `<div class="guide" style="grid-column:1/-1">
    <p style="margin:0 0 12px;">Getting good results takes a few iterations. Start by adding a pet, building up references, and adding some negatives. Run a short test scan, review the results, refine, and repeat until you're satisfied. Then run the full backfill.</p>
    <div class="guide-steps">
      <div class="guide-step"><div class="guide-step-num">1</div><div class="guide-step-body"><div class="guide-step-title">Add your pet</div><div class="guide-step-desc"><strong>Import from Immich</strong>: use this if Immich already recognizes your pet as a person from its own face detection. This is ideal when the person in Immich contains only photos of that pet, for example if you tagged them manually and are confident the assignments are correct. The tagger does not remove or correct existing Immich face assignments, so any misidentified photos already tagged in Immich will stay tagged. If Immich's recognition was noisy, consider adding your pet manually instead.<ol><li>Click <strong>↓ Import from Immich</strong> in the sidebar</li><li>Find and click your pet in the grid</li><li>Enter a short description (e.g. <code>orange tabby cat</code>) and an optional date range</li><li>Click <strong>Import</strong>. Up to 20 reference photos are imported automatically.</li></ol><strong>Add manually</strong>: use this if Immich doesn't know your pet yet.<ol><li>Click <strong>+ Add pet</strong>, fill in the name, a short description (e.g. <code>black labrador dog</code>), and an optional date range</li><li>Click <strong>Create</strong></li></ol>The description is used by Immich's CLIP model to find the first batch of candidate photos. Keep it short: 2–4 descriptive keywords.</div></div></div>
      <div class="guide-step"><div class="guide-step-num">2</div><div class="guide-step-body"><div class="guide-step-title">Add reference photos</div><div class="guide-step-desc">References are what the classifier learns from. Quality matters more than quantity.<ol><li>Select your pet in the sidebar and click <strong>Find references</strong></li><li>Browse the results. They are ranked by visual similarity to your existing refs, or to your description if no refs exist yet.</li><li>Aim for 20–30 to start; results improve up to around 50. For each photo:<ul><li><strong>Your pet's name</strong> (one button per pet): clear, close-up shot, that pet is the only subject. If the photo shows another of your pets, click that pet's button instead.</li><li><strong>Ignore</strong>: blurry, distant, another person or animal visible alongside your pet, or a look-alike that is not yours. Ignored photos won't appear again. You can see and restore them in <strong>Ignored</strong> in the sidebar.</li><li><strong>Not a pet</strong>: photos that could confuse the classifier. Empty rooms, other species, ambiguous shots. Around 50 is enough.</li></ul></li></ol>If you already know a specific photo you want to use, click <strong>Add manually</strong> and paste the Immich photo URL or asset ID directly.</div></div></div>
      <div class="guide-step"><div class="guide-step-num">3</div><div class="guide-step-body"><div class="guide-step-title">Add "not a pet" samples</div><div class="guide-step-desc">These teach the classifier what not to tag: empty rooms, other animals of a different species, ambiguous shots with no clear subject. Without them, the classifier will tag almost anything.<ol><li>Click <strong>Not a pet</strong> in the sidebar, then <strong>Find candidates</strong> to automatically surface more photos that might confuse the classifier</li><li>Select the relevant ones and click <strong>Not a pet</strong></li><li>To add a specific photo directly, click <strong>Add manually</strong> and paste its Immich URL or asset ID</li></ol></div></div></div>
      <div class="guide-step"><div class="guide-step-num">4</div><div class="guide-step-body"><div class="guide-step-title">Run a test scan</div><div class="guide-step-desc">Start with a recent date so the scan covers fewer photos, making it quicker to review and refine before committing to a full backfill.<ol><li>In the scan panel at the bottom of the sidebar, set the <strong>From</strong> date 1–2 weeks back</li><li>Click <strong>Scan</strong> and wait for the results</li><li>If <strong>Review N low confidence</strong> appears in the results, click it to see photos the classifier identified as a match but wasn't fully confident about.</li><li>Go through them: add correctly identified ones as references, and click <strong>Ignore</strong> on the rest. Ignored photos won't appear in future results.</li></ol></div></div></div>
      <div class="guide-step"><div class="guide-step-num">5</div><div class="guide-step-body"><div class="guide-step-title">Iterate</div><div class="guide-step-desc">Repeat steps 2–4 a couple of times. Each round of added references and negatives improves accuracy. Results typically stabilize after 2–3 iterations.<br><br>If you want a more precise read on accuracy than eyeballing the scan results, use <strong>Tagging accuracy</strong> (📊 icon in the sidebar, or <code>/accuracy.html</code>): it dry-runs classification over a date range and compares it against your existing Immich tags, showing recall and false-positive rates broken down by pet, photo/video, and detection confidence, and lets you test different thresholds without waiting for a real scan.</div></div></div>
      <div class="guide-step"><div class="guide-step-num">6</div><div class="guide-step-body"><div class="guide-step-title">Run the full backfill</div><div class="guide-step-desc">Once you're happy with the accuracy on the test window:<ol><li>Set the scan date to the earliest date you want to tag. A good starting point is the date you got your pet.</li><li>Click <strong>Scan</strong> to process all photos in that range</li></ol>After that, the background poller runs every hour (<code>POLL_INTERVAL</code>) and tags new photos automatically. Your pets appear in Immich's <strong>People</strong> section.</div></div></div>
    </div>
  </div>`;
}

// The center shows the active pet's refs or list's photos ('list'), finder results ('results'),
// or the guide ('guide').
function setCenterView(view) {
  document.getElementById('centerPanel').dataset.view = view;
}

// Switching the sidebar selection: confirm dropping unassigned picks, then reset finder modes.
function leaveSelection() {
  if (selectedCrops.size > 0) {
    const ok = confirm(`You have ${selectedCrops.size} selected photo${selectedCrops.size !== 1 ? 's' : ''} not yet assigned. Switch anyway?`);
    if (!ok) return false;
  }
  resetResults();
  return true;
}

// Drops the current finder: stops its progress poller, clears its results and selection,
// and bumping resultsGen makes any response still in flight see it is stale and not paint
// over what is shown now.
function resetResults() {
  resultsGen++;
  resultsMode = null;
  if (progressTimer) { clearInterval(progressTimer); progressTimer = null; }
  document.getElementById('scanFilterBtns').style.display = 'none';
  document.getElementById('resultsLabel').textContent = '';
  document.getElementById('photoGrid').innerHTML = '';
  selectedCrops.clear(); lastClickedKey = null; updateSelUI();
}

// Shows a finder's server-side progress in its loading message, until the finder ends
// (the caller clears the returned timer) or a newer one starts (resultsGen moves on).
function pollProgress(url, gen) {
  const timer = progressTimer = setInterval(async () => {
    if (resultsGen !== gen) { clearInterval(timer); return; }
    try {
      const p = await api(url);
      const el = document.getElementById('resultsLoadMsg');
      if (!el || resultsGen !== gen) return;
      if (p.total > 0) el.textContent = `Loading ${Math.round(p.current / p.total * 100)}%…`;
      else if (p.running) el.textContent = 'Loading…';
    } catch(_) {}
  }, 1000);
  return timer;
}

// Centered empty state for the results grid. With a title it also gets the paw icon.
function emptyState(title, sub) {
  return `<div class="empty" style="grid-column:1/-1;height:200px;">${title ? `<div class="empty-icon">🐾</div><div class="empty-title">${title}</div>` : ''}<div class="empty-sub">${sub}</div></div>`;
}

// Every finder starts here. mode is 'suggest' | 'borderline' | 'negCandidates' | 'lowConf' |
// 'pickCrop'. Returns the generation to compare with resultsGen after each await.
function startResults(mode, labelText, loadingText = null) {
  resetResults();
  resultsMode = mode;
  setCenterView('results');
  document.getElementById('resultsLabel').textContent = labelText;
  if (loadingText != null) document.getElementById('photoGrid').innerHTML = `<div class="loading" id="resultsLoadMsg" style="grid-column:1/-1">${loadingText}</div>`;
  return resultsGen;
}

// The center header serves the active pet or the active list; data-kind picks its buttons.
function showCenterHeader(title, kind) {
  document.getElementById('centerTitle').textContent = title;
  const bar = document.getElementById('centerBar');
  bar.dataset.kind = kind;
  bar.style.display = '';
  setCenterView('list');
}

async function selectPet(name) {
  if (activePet?.name === name) { setCenterView('list'); return; }
  if (!leaveSelection()) return;
  setSelection(pets.find(p => p.name === name));
  renderSidebar();
  showCenterHeader(name, 'pet');
  await loadRefs(name);
}

async function selectList(kind) {
  if (activeList === kind) { setCenterView('list'); return; }
  if (!leaveSelection()) return;
  setSelection(null, kind);
  renderSidebar();
  showCenterHeader(TILE_LISTS[kind].title, kind);
  document.getElementById('listViewBtn').textContent = TILE_LISTS[kind].title;
  document.getElementById('listGrid').innerHTML = '<div class="loading">Loading…</div>';
  await loadTileList(kind);
}

async function loadRefs(name) {
  const grid = document.getElementById('listGrid');
  grid.innerHTML = '<div class="loading">Loading…</div>';
  try {
    const d = await api(`/api/pets/${encodeURIComponent(name)}/assets`);
    if (activePet?.name !== name) return;
    refsItems = d.assets;
    renderRefs(d.assets);
  } catch(e) { grid.innerHTML = '<div class="empty" style="grid-column:1/-1"><div class="empty-sub">Error loading refs</div></div>'; }
}

function renderRefs(assets) {
  const grid = document.getElementById('listGrid');
  document.getElementById('listViewBtn').textContent = `References (${assets.length})`;
  if (!assets.length) { grid.innerHTML = '<div class="empty" style="grid-column:1/-1;height:160px;"><div class="empty-sub">No references yet.<br>Click "Find references" to add some.</div></div>'; return; }
  renderTileGrid(grid, assets, a => `removeRef('${a.id}', ${a.crop_idx != null ? a.crop_idx : 'null'})`);
}

// The .ref-thumb grid of a pet's refs and of the tile lists. removeCall(a) is the onclick
// code of a tile's remove button. Keeps the scroll position, so removing a tile doesn't jump.
function renderTileGrid(grid, items, removeCall) {
  const scrollTop = grid.scrollTop;
  grid.innerHTML = items.map(a => `
    <div class="ref-thumb">
      <a href="${immichUrl}/photos/${a.id}" target="_blank" rel="noopener" title="Open in Immich">
        <img src="${a.thumb}" loading="lazy" onerror="this.style.opacity=0.2">
      </a>
      <button class="ref-remove" onclick="${removeCall(a)}" title="Remove">✕</button>
    </div>`).join('');
  grid.scrollTop = scrollTop;
}

async function removeRef(assetId, cropIdx = null) {
  if (!activePet) return;
  const pet = activePet;
  try {
    const url = cropIdx != null
      ? `/api/pets/${encodeURIComponent(pet.name)}/assets/${assetId}?crop_idx=${cropIdx}`
      : `/api/pets/${encodeURIComponent(pet.name)}/assets/${assetId}`;
    await api(url, { method: 'DELETE' });
    refsItems = refsItems.filter(r => {
      if (r.id !== assetId) return true;
      if (cropIdx != null) return r.crop_idx !== cropIdx;
      return false;
    });
    renderRefs(refsItems);
    await loadPets();  // only the ref count changed, refresh it in the sidebar
    toast('Removed');
  } catch(e) { toast('Error: ' + e.message, 'error'); }
}

// Adds crops ({asset_id, crop_idx, bbox}) as refs of petName. The POST replaces the whole
// list, so merge with the server's current refs, deduped per asset+crop, then refresh.
// Shared by the finder grid and Add by ID so both dedupe the same way.
async function addRefs(petName, crops) {
  const existing = await api(`/api/pets/${encodeURIComponent(petName)}/assets`);
  const existingCrops = existing.assets.map(a => ({ asset_id: a.id, crop_idx: a.crop_idx, bbox: a.bbox }));
  const seen = new Set();
  const merged = [...existingCrops, ...crops].filter(c => {
    const k = c.crop_idx != null ? `${c.asset_id}_${c.crop_idx}` : c.asset_id;
    if (seen.has(k)) return false;
    seen.add(k); return true;
  });
  await api(`/api/pets/${encodeURIComponent(petName)}/assets`, { method: 'POST', body: { assets: merged } });
  if (activePet?.name === petName) await loadRefs(petName);
  await refreshState();
}

// Adds the selected photos as refs of any pet, not only the active one, so a photo of
// another pet found in this pet's list can go straight to that pet (see #64).
async function assignSelectedTo(petName) {
  if (!selectedCrops.size) return;
  const newCrops = [...selectedCrops.values()];
  try {
    await addRefs(petName, newCrops);
    selectedCrops.forEach((_, key) => {
      const el = document.getElementById('th-' + key);
      if (el) { el.classList.remove('selected'); el.classList.add('is-ref'); }
    });
    selectedCrops.clear(); updateSelUI();
    dropScanLowConf(newCrops.map(c => c.asset_id));
    toast(`Added ${newCrops.length} to ${petName}`, 'success');
  } catch(e) { toast('Error: ' + e.message, 'error'); }
}

function renderPetBtns() {
  const el = document.getElementById('petBtns');
  el.innerHTML = pets.map(p => `<button class="btn btn-primary" title="Add to ${escapeHtml(p.name)}: clear, close-up shot, your pet is the only subject.">${escapeHtml(p.name)}</button>`).join('');
  [...el.children].forEach((btn, i) => { btn.onclick = () => assignSelectedTo(pets[i].name); });
}

// ---------------------------------------------------------------------------
// Photo grid rendering helpers
// ---------------------------------------------------------------------------

function getCropData(el) {
  const cropIdx = el.dataset.cropIdx !== undefined && el.dataset.cropIdx !== '' ? parseInt(el.dataset.cropIdx) : null;
  const bbox = el.dataset.bbox ? JSON.parse(el.dataset.bbox) : null;
  return { asset_id: el.dataset.assetId, crop_idx: cropIdx, bbox };
}

function renderPhotoItems(a, thr) {
  const badge = a.score != null
    ? `<div class="score-badge ${a.score < thr ? 'score-low' : 'score-ok'}">${Math.round(a.score * 100)}%</div>`
    : '';
  const title = a.pet_name
    ? `${fmtDate(a.date)} · ${Math.round(a.score * 100)}% ${a.pet_name}`
    : `${a.filename || ''} · ${fmtDate(a.date)}`;
  const makeItem = (key, src, cropIdx, bbox) => {
    const cropIdxAttr = cropIdx != null ? `data-crop-idx="${cropIdx}"` : '';
    const bboxAttr = bbox ? `data-bbox='${JSON.stringify(bbox)}'` : '';
    return `<div class="photo-thumb" id="th-${key}" data-asset-id="${a.id}" ${cropIdxAttr} ${bboxAttr}
      onclick="toggleSelect(event,'${key}')" title="${escapeHtml(title)}">
      <img src="${src}" loading="lazy" onerror="this.src='data:image/svg+xml,<svg/>'">
      <a class="photo-open" href="${immichUrl}/photos/${a.id}" target="_blank" rel="noopener" onclick="event.stopPropagation()">⤢</a>
      <div class="photo-check">✓</div>
      ${badge}
    </div>`;
  };
  if (a.crops && a.crops.length > 0) {
    return a.crops.map(c => makeItem(`${a.id}_${c.crop_idx}`, `/api/crop/${a.id}?bbox=${c.bbox.join(',')}`, c.crop_idx, c.bbox));
  }
  return [makeItem(a.id, a.thumb, null, a.bbox || null)];
}

function markGridItems(assets) {
  const refKeys = new Set(refsItems.map(r => r.crop_idx != null ? `${r.id}_${r.crop_idx}` : r.id));
  assets.forEach(a => {
    const keys = (a.crops && a.crops.length > 0) ? a.crops.map(c => `${a.id}_${c.crop_idx}`) : [a.id];
    keys.forEach(key => {
      const el = document.getElementById('th-' + key);
      if (!el) return;
      if (refKeys.has(key) || refKeys.has(a.id)) el.classList.add('is-ref');
    });
  });
  markNegTiles();
}

// "Not a pet" entries are {asset_id, bbox}: bbox null covers every crop of the photo, a bbox
// only that crop. Mirrors data.crop_entry_matches on the server (same 0.02 tolerance).
function bboxesMatch(a, b) {
  return !!a && !!b && a.length === b.length && a.every((v, i) => Math.abs(v - b[i]) <= 0.02);
}
function isNegCrop(assetId, bbox) {
  return negEntries.some(e => e.asset_id === assetId && (e.bbox == null || bboxesMatch(e.bbox, bbox)));
}
function markNegTiles() {
  document.querySelectorAll('#photoGrid .photo-thumb').forEach(el => {
    const c = getCropData(el);
    el.classList.toggle('is-neg', isNegCrop(c.asset_id, c.bbox));
  });
}

// ---------------------------------------------------------------------------
// Ref suggestions
// ---------------------------------------------------------------------------

function viewFindRefs() {
  if (!activePet) return;
  if (activePet.ref_count > 0) viewBorderline();
  else viewSuggestions();
}

// ---------------------------------------------------------------------------
// Add by ID / link (refs and negatives)
// ---------------------------------------------------------------------------

let _addByIdMode = 'ref'; // 'ref' | 'neg'

function parseAssetId(input) {
  const s = (input || '').trim();
  const UUID = '[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}';
  const afterPhotos = s.match(new RegExp('photos/(' + UUID + ')', 'i'));
  if (afterPhotos) return afterPhotos[1];
  const any = s.match(new RegExp(UUID, 'i'));
  return any ? any[0] : null;
}

function _openAddByIdModal(mode) {
  _addByIdMode = mode;
  document.getElementById('addByIdInput').value = '';
  document.getElementById('addByIdTextarea').value = '';
  clearModalError('addByIdError');
  const example = 'e.g. a1b2c3d4-11aa-22bb-33cc-4d5e6f7a8b9c';
  const singleInput = document.getElementById('addByIdInput');
  const multiInput = document.getElementById('addByIdTextarea');
  if (mode === 'neg') {
    document.getElementById('addByIdTitle').textContent = 'Add "not a pet" by ID or link';
    document.getElementById('addByIdLabel').textContent = 'Immich photo links or asset IDs, one per line';
    document.getElementById('addByIdHint').textContent = `Paste one or more Immich photo URLs or asset IDs, one per line (${example}). They'll be added directly to "not a pet".`;
    singleInput.style.display = 'none';
    multiInput.style.display = 'block';
  } else {
    document.getElementById('addByIdTitle').textContent = 'Add reference by ID or link';
    document.getElementById('addByIdLabel').textContent = 'Immich photo link or asset ID';
    document.getElementById('addByIdHint').textContent = `Paste an Immich photo URL or just the asset ID (${example}). If one animal is detected it is added immediately; multiple crops let you pick the right one.`;
    singleInput.style.display = 'block';
    multiInput.style.display = 'none';
  }
  document.getElementById('addByIdModal').classList.add('open');
  setTimeout(() => (mode === 'neg' ? multiInput : singleInput).focus(), 100);
}

function openAddById() { if (activePet) _openAddByIdModal('ref'); }
function openAddNegById() { _openAddByIdModal('neg'); }

function closeAddById() {
  document.getElementById('addByIdModal').classList.remove('open');
  clearModalError('addByIdError');
}

// Looking up a photo's crops loads the models and can take a few seconds, so the Add
// button shows it is working and can't be clicked twice.
async function submitAddById() {
  const btn = document.getElementById('addByIdSubmit');
  if (btn.disabled) return;
  btn.disabled = true; btn.textContent = 'Adding…';
  try { await _submitAddById(); }
  finally { btn.disabled = false; btn.textContent = 'Add'; }
}

async function _submitAddById() {
  clearModalError('addByIdError');

  if (_addByIdMode === 'neg') {
    const lines = document.getElementById('addByIdTextarea').value.split('\n').map(l => l.trim()).filter(Boolean);
    const ids = [...new Set(lines.map(parseAssetId).filter(Boolean))];
    if (ids.length === 0) { modalError('addByIdError', 'Could not find any asset IDs. Paste Immich photo links or IDs, one per line.'); return; }
    try {
      const result = await api('/api/negatives/batch', { method: 'POST', body: { asset_ids: ids } });
      closeAddById();
      await loadNegatives();
      const invalidCount = (result.invalid_ids || []).length;
      const msg = invalidCount > 0
        ? `Added ${result.added} to "not a pet" (${invalidCount} not found)`
        : `Added ${result.added} to "not a pet"`;
      toast(msg, invalidCount > 0 ? 'warn' : 'success');
    } catch(e) {
      modalError('addByIdError', e.message || 'Could not load assets');
    }
    return;
  }

  // ref mode
  const id = parseAssetId(document.getElementById('addByIdInput').value);
  if (!id) { modalError('addByIdError', 'Could not find an asset ID. Paste an Immich photo link or the bare ID.'); return; }
  if (!activePet) return;
  const pet = activePet;
  try {
    const a = await api(`/api/asset/${id}/crops`);
    const crops = a.crops || [];
    if (crops.length <= 1) {
      // 0 or 1 crop: add immediately, no picker needed
      const newCrop = crops.length === 1
        ? { asset_id: id, crop_idx: crops[0].crop_idx, bbox: crops[0].bbox }
        : { asset_id: id, crop_idx: null, bbox: null };
      await addRefs(pet.name, [newCrop]);
      closeAddById();
      toast(`Added to ${pet.name}`, 'success');
    } else {
      // Multiple crops: load into grid so user picks the right animal
      closeAddById();
      startResults('pickCrop', `${crops.length} pets detected. Select the one to add as reference`);
      document.getElementById('photoGrid').innerHTML = renderPhotoItems(a, 0.8).join('');
      markGridItems([a]);
    }
  } catch(e) {
    modalError('addByIdError', e.message || 'Could not load asset');
  }
}

async function viewSuggestions() {
  if (!activePet) return;
  const pet = activePet;
  if (!pet.description) { toast('Edit this pet and add a description to use this feature', 'error'); return; }
  const myGen = startResults('suggest', 'Finding references…', 'Finding similar photos… this may take a moment');
  const grid = document.getElementById('photoGrid');
  const label = document.getElementById('resultsLabel');
  try {
    const d = await api(`/api/pets/${encodeURIComponent(pet.name)}/suggestions`);
    if (resultsGen !== myGen) return;
    label.textContent = `${d.assets.length} photo${d.assets.length !== 1 ? 's' : ''} similar to ${pet.name}`;
    if (!d.assets.length) {
      grid.innerHTML = emptyState('No suggestions found', 'Add more refs or broaden the date range');
      return;
    }
    grid.innerHTML = d.assets.flatMap(a => renderPhotoItems(a, 0.8)).join('');
    markGridItems(d.assets);
  } catch(e) {
    if (resultsGen !== myGen) return;
    label.textContent = 'Failed to load suggestions';
    grid.innerHTML = emptyState(null, e.message);
    toast('Suggestions error: ' + e.message, 'error');
  }
}

async function viewBorderline() {
  if (!activePet || !activePet.ref_count) return;
  const petName = activePet.name;
  const myGen = startResults('borderline', 'Finding references…', 'Loading…');
  const grid = document.getElementById('photoGrid');
  const label = document.getElementById('resultsLabel');

  const timer = pollProgress(`/api/pets/${encodeURIComponent(petName)}/borderline/progress`, myGen);

  try {
    const d = await api(`/api/pets/${encodeURIComponent(petName)}/borderline`);
    clearInterval(timer);
    if (resultsGen !== myGen) return;
    label.textContent = `${d.assets.length} photo${d.assets.length !== 1 ? 's' : ''} ${petName} might be missing. Add good ones as refs to improve accuracy.`;
    if (!d.assets.length) {
      grid.innerHTML = emptyState('No missed photos found', 'The classifier is either very confident or not finding this pet at all');
      return;
    }
    const thr = d.threshold ?? 0.8;
    grid.innerHTML = d.assets.flatMap(a => renderPhotoItems(a, thr)).join('');
    markGridItems(d.assets);
  } catch(e) {
    clearInterval(timer);
    if (resultsGen !== myGen) return;
    label.textContent = 'Failed to load missed photos';
    grid.innerHTML = emptyState(null, e.message);
    toast('Error: ' + e.message, 'error');
  }
}

// ---------------------------------------------------------------------------

function toggleSelect(e, key) {
  const el = document.getElementById('th-' + key); if (!el) return;
  if (el.classList.contains('is-ref')) return;
  if (el.classList.contains('is-neg')) return;
  if (e.shiftKey && lastClickedKey && lastClickedKey !== key) {
    const thumbs = [...document.querySelectorAll('#photoGrid .photo-thumb')];
    const fromEl = document.getElementById('th-' + lastClickedKey);
    const fromIdx = thumbs.indexOf(fromEl), toIdx = thumbs.indexOf(el);
    if (fromIdx !== -1 && toIdx !== -1) {
      const lo = Math.min(fromIdx, toIdx), hi = Math.max(fromIdx, toIdx);
      for (let i = lo; i <= hi; i++) {
        if (thumbs[i].classList.contains('is-ref') || thumbs[i].classList.contains('is-neg')) continue;
        const tkey = thumbs[i].id.slice(3);
        if (!selectedCrops.has(tkey)) { selectedCrops.set(tkey, getCropData(thumbs[i])); thumbs[i].classList.add('selected'); }
      }
    }
  } else {
    if (selectedCrops.has(key)) { selectedCrops.delete(key); el.classList.remove('selected'); }
    else { selectedCrops.set(key, getCropData(el)); el.classList.add('selected'); }
    lastClickedKey = key;
  }
  updateSelUI();
}

function updateSelUI() {
  const n = selectedCrops.size;
  document.getElementById('selCount').textContent = n ? `${n} selected` : '';
  const showPets = n && resultsMode !== 'negCandidates' && pets.length;
  if (showPets) renderPetBtns();
  document.getElementById('petBtns').style.display = showPets ? 'flex' : 'none';
  document.getElementById('ignoreBtn').style.display = n ? '' : 'none';
  document.getElementById('addNegBtn').style.display = n ? '' : 'none';
}

async function ignoreSelected() {
  if (!selectedCrops.size) return;
  const ids = [...new Set([...selectedCrops.values()].map(c => c.asset_id))];
  try {
    await api('/api/skipped', { method: 'POST', body: { asset_ids: ids } });
    ids.forEach(id => document.querySelectorAll(`[data-asset-id="${id}"]`).forEach(el => el.remove()));
    selectedCrops.clear(); updateSelUI();
    dropScanLowConf(ids);
    loadIgnored();
    toast(`Ignored ${ids.length} photo${ids.length !== 1 ? 's' : ''}. Won't appear again.`, 'success');
  } catch(e) { toast('Error: ' + e.message, 'error'); }
}

// ---------------------------------------------------------------------------
// Negatives
// ---------------------------------------------------------------------------

// Tile lists shown in the center from the sidebar rows, sharing rendering and removal.
const TILE_LISTS = {
  neg: {
    path: '/api/negatives', title: 'Not a pet', glyph: '<svg viewBox="0 0 16 16" width="14" height="14" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" aria-hidden="true"><path d="M4 4l8 8M12 4l-8 8"/></svg>', clearBtn: 'clearNegsBtn', label: 'not a pet',
    // The ✕ removes exactly that entry: one crop, or the whole-photo entry.
    removeQuery: a => '?bbox=' + (a.bbox ? a.bbox.join(',') : 'whole'),
    onLoad: assets => { negEntries = assets.map(a => ({ asset_id: a.id, bbox: a.bbox ?? null })); markNegTiles(); },
  },
  ignored: {
    path: '/api/skipped', title: 'Ignored', glyph: '<svg viewBox="0 0 16 16" width="14" height="14" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" aria-hidden="true"><circle cx="8" cy="8" r="6"/><path d="M3.8 12.2l8.4-8.4"/></svg>', label: 'ignored',
    empty: 'No ignored photos', newestFirst: true,
    removedMsg: 'Removed from ignored. It can appear in suggestions again.',
  },
};

async function loadTileList(kind) {
  const L = TILE_LISTS[kind];
  const grid = document.getElementById('listGrid');
  try {
    const d = await api(L.path);
    const assets = L.newestFirst ? [...d.assets].reverse() : d.assets;
    L.onLoad?.(d.assets);
    listCounts[kind] = d.assets.length;
    renderListRows();
    if (L.clearBtn) document.getElementById(L.clearBtn).style.display = d.assets.length ? '' : 'none';
    if (activeList !== kind) return;
    document.getElementById('listViewBtn').textContent = `${L.title} (${d.assets.length})`;
    if (!assets.length) { grid.innerHTML = L.empty ? `<div class="tile-empty">${L.empty}</div>` : ''; return; }
    renderTileGrid(grid, assets, a => `removeTile('${kind}','${a.id}','${L.removeQuery ? L.removeQuery(a) : ''}')`);
  } catch(e) {
    console.warn('loadTileList ' + kind + ':', e);
    if (activeList === kind) grid.innerHTML = `<div class="empty" style="grid-column:1/-1"><div class="empty-sub">Error loading ${escapeHtml(L.label)}</div></div>`;
  }
}

function loadNegatives() { return loadTileList('neg'); }
function loadIgnored() { return loadTileList('ignored'); }

async function removeTile(kind, id, query = '') {
  const L = TILE_LISTS[kind];
  try {
    await api(`${L.path}/${id}${query}`, { method: 'DELETE' });
    await loadTileList(kind);
    toast(L.removedMsg || `Removed from "${L.label}"`);
  } catch(e) { toast('Error: ' + e.message, 'error'); }
}

async function addSelectedAsNegatives() {
  if (!selectedCrops.size) return;
  const entries = [...selectedCrops.values()].map(c => ({ asset_id: c.asset_id, bbox: c.bbox }));
  const assetIds = [...new Set(entries.map(e => e.asset_id))];
  try {
    await api('/api/negatives', { method: 'POST', body: { entries } });
    selectedCrops.forEach((_, key) => document.getElementById('th-' + key)?.classList.remove('selected'));
    selectedCrops.clear(); updateSelUI();
    dropScanLowConf(assetIds);
    await loadNegatives();  // reloads negEntries and marks the matching tiles
    toast('Added to "not a pet"', 'success');
  } catch(e) { toast('Error: ' + e.message, 'error'); }
}

async function viewNegCandidates() {
  const myGen = startResults('negCandidates', 'Finding candidates…', 'Loading…');
  const grid = document.getElementById('photoGrid');
  const label = document.getElementById('resultsLabel');

  const timer = pollProgress('/api/suggestions/negatives/progress', myGen);

  try {
    const d = await api('/api/suggestions/negatives');
    clearInterval(timer);
    if (resultsGen !== myGen) return;
    label.textContent = `${d.assets.length} candidate${d.assets.length !== 1 ? 's' : ''} for "not a pet"`;
    if (!d.assets.length) {
      grid.innerHTML = emptyState('No candidates found', 'No uncertain photos in this random sample. Click "Find candidates" again to try another sample.');
      return;
    }
    const thr = d.threshold || 0.8;
    grid.innerHTML = d.assets.flatMap(a => renderPhotoItems(a, thr)).join('');
    markNegTiles();
  } catch(e) {
    clearInterval(timer);
    if (resultsGen !== myGen) return;
    label.textContent = 'Failed to load candidates';
    grid.innerHTML = emptyState(null, e.message);
    toast('Error: ' + e.message, 'error');
  }
}

async function clearAllRefs() {
  if (!activePet) return;
  const pet = activePet;
  if (!confirm(`Remove all reference photos for ${pet.name} from Pet Tagger? This will not affect Immich.`)) return;
  try {
    await api(`/api/pets/${encodeURIComponent(pet.name)}/refs`, { method: 'DELETE' });
    refsItems = [];
    await loadRefs(pet.name);
    await refreshState();
    toast('All refs cleared', 'success');
  } catch(e) { toast('Error: ' + e.message, 'error'); }
}

async function clearAllNegatives() {
  if (!confirm(`Remove all "not a pet" photos from Pet Tagger? This will not affect Immich.`)) return;
  try {
    await api('/api/negatives/all', { method: 'DELETE' });
    negEntries = [];
    await loadNegatives();
    toast('All "not a pet" photos cleared', 'success');
  } catch(e) { toast('Error: ' + e.message, 'error'); }
}

// ---------------------------------------------------------------------------
// Poll status
// ---------------------------------------------------------------------------

function fmtDate(iso) {
  if (!iso) return '';
  const [y, m, d] = iso.slice(0, 10).split('-');
  return new Date(+y, m - 1, +d).toLocaleDateString();
}


// ---------------------------------------------------------------------------
// Scan timestamp
// ---------------------------------------------------------------------------

function prefillScanUntil() {
  const el = document.getElementById('scanUntil');
  if (!el.value) el.value = new Date().toISOString().slice(0, 10);
}

async function loadScanResult() {
  try { showScanResult(await api('/api/scan/result')); } catch(_) {}
}

function showScanResult(r) {
  const el = document.getElementById('scanResult');
  const stopBtn = document.getElementById('stopScanBtn');
  if (!r || r.status === 'none') { el.style.display = 'none'; stopBtn.style.display = 'none'; return; }
  el.className = 'scan-result';
  el.style.display = '';
  const stat = (label, val, cls) => `<div class="poll-stat"><span class="poll-stat-label">${label}</span><span class="poll-stat-val ${val > 0 ? cls : ''}">${val}</span></div>`;
  if (r.status === 'running') {
    stopBtn.style.display = '';
    const dateStr = r.current_date ? new Date(r.current_date + 'T00:00:00').toLocaleDateString() : '';
    const c = r.counts || {};
    el.innerHTML = '<div class="scan-result-header">Scanning…</div>' +
      (dateStr ? `<div style="font-size:11px;color:var(--text3);margin-top:4px;">${dateStr}</div>` : '') +
      '<div class="poll-stats" style="margin-top:6px;">' +
      stat('Tagged', c.added || 0, 'nonzero-good') +
      stat('Low conf.', c.low_confidence || 0, 'nonzero-warn') +
      stat('Other', c.unknown || 0, '') +
      stat('Already tagged', c.already_tagged || 0, '') +
      (c.failed > 0 ? stat('Failed', c.failed, 'nonzero-bad') : '') +
      '</div>';
    return;
  }
  stopBtn.style.display = 'none';
  if (r.status === 'stopped') {
    el.innerHTML = '<div class="scan-result-header">Scan stopped</div>';
    return;
  }
  if (r.status === 'error') {
    el.innerHTML = `<div class="scan-result-header">Scan failed</div><div style="font-size:11px;color:var(--danger);margin-top:4px;">${r.error || ''}</div>`;
    return;
  }
  if (r.counts) {
    const c = r.counts;
    el.innerHTML = '<div class="scan-result-header">Scan result</div>' +
      (c.fetched === 0 ? '<div style="font-size:11px;color:var(--text3);margin-top:4px;">No photos found in this date range.</div>' : '') +
      '<div class="poll-stats" style="margin-top:6px;">' +
      (c.fetched != null ? stat('Photos scanned', c.fetched, '') : '') +
      stat('Tagged', c.added, 'nonzero-good') +
      stat('Low conf.', c.low_confidence, 'nonzero-warn') +
      stat('Other', c.unknown, '') +
      stat('Out of range', c.out_of_range, '') +
      stat('Already tagged', c.already_tagged, '') +
      (c.failed > 0 ? stat('Failed', c.failed, 'nonzero-bad') : '') +
      (c.no_thumb > 0 ? stat('No thumb', c.no_thumb, 'nonzero-warn') : '') +
      '</div>' +
      (c.low_confidence > 0 ? `<button class="btn" style="font-size:11px;margin-top:8px;width:100%;" onclick="viewScanLowConf()">Review ${c.low_confidence} low confidence</button>` : '');
  }
}

async function viewScanLowConf() {
  scanLowConfFilter = null;
  const myGen = startResults('lowConf', 'Loading…', 'Loading low confidence results…');
  const grid = document.getElementById('photoGrid');
  const label = document.getElementById('resultsLabel');
  try {
    const d = await api('/api/scan/low-confidence');
    if (resultsGen !== myGen) return;
    scanLowConfAssets = d.assets;
    scanLowConfThreshold = d.threshold ?? 0.8;
    if (!d.assets.length) {
      label.textContent = 'No low confidence results';
      grid.innerHTML = emptyState(null, 'All results were confident or unknown');
      return;
    }
    renderScanLowConfFilterBtns();
    renderScanLowConf();
  } catch(e) {
    if (resultsGen !== myGen) return;
    label.textContent = 'Failed to load';
    grid.innerHTML = emptyState(null, e.message);
  }
}

function renderScanLowConfFilterBtns() {
  const el = document.getElementById('scanFilterBtns');
  const counts = new Map();
  scanLowConfAssets.forEach(a => counts.set(a.pet_name, (counts.get(a.pet_name) || 0) + 1));
  const petNames = pets.map(p => p.name).filter(n => counts.has(n));
  el.innerHTML = [
    `<button class="btn ${scanLowConfFilter === null ? 'btn-primary' : ''}" data-filter-pet="">All (${scanLowConfAssets.length})</button>`,
    ...petNames.map(n => `<button class="btn ${scanLowConfFilter === n ? 'btn-primary' : ''}" data-filter-pet="${n}">${n} (${counts.get(n)})</button>`)
  ].join('');
  [...el.children].forEach(btn => { btn.onclick = () => { scanLowConfFilter = btn.dataset.filterPet || null; renderScanLowConfFilterBtns(); renderScanLowConf(); }; });
  el.style.display = petNames.length > 1 ? 'flex' : 'none';
}

// Answered low confidence results (ignored, assigned, not a pet) leave the list, so
// switching the pet filter re-renders without them and the filter counts stay right.
function dropScanLowConf(assetIds) {
  if (resultsMode !== 'lowConf') return;
  const drop = new Set(assetIds);
  scanLowConfAssets = scanLowConfAssets.filter(a => !drop.has(a.id));
  if (scanLowConfFilter !== null && !scanLowConfAssets.some(a => a.pet_name === scanLowConfFilter)) scanLowConfFilter = null;
  renderScanLowConfFilterBtns();
  renderScanLowConf();
}

function renderScanLowConf() {
  const grid = document.getElementById('photoGrid');
  const label = document.getElementById('resultsLabel');
  selectedCrops.clear(); lastClickedKey = null; updateSelUI();
  const assets = scanLowConfFilter === null ? scanLowConfAssets : scanLowConfAssets.filter(a => a.pet_name === scanLowConfFilter);
  label.textContent = `${assets.length} low confidence result${assets.length !== 1 ? 's' : ''}`;
  grid.innerHTML = assets.flatMap(a => renderPhotoItems(a, scanLowConfThreshold)).join('');
  markNegTiles();
}

async function applyTimestamp() {
  const val = document.getElementById('scanDate').value;
  if (!val) { toast('Pick a date first', 'error'); return; }
  const untilVal = document.getElementById('scanUntil').value || null;
  try {
    await api('/api/scan', { method: 'POST', body: { scan_since: val, scan_until: untilVal } });
    showScanResult({ status: 'running' });
    const iv = setInterval(async () => {
      try {
        const r = await api('/api/scan/result');
        showScanResult(r);
        if (r.status !== 'running') { clearInterval(iv); }
      } catch(_) {}
    }, 2000);
  } catch(e) {
    toast(e.message, 'error');
  }
}

async function stopScan() {
  try {
    await api('/api/scan/stop', { method: 'POST' });
    showScanResult({ status: 'stopped' });
  } catch(e) {
    toast(e.message, 'error');
  }
}

// ---------------------------------------------------------------------------
// Modals
// ---------------------------------------------------------------------------

function modalError(id, msg) { document.getElementById(id).textContent = msg; }
function clearModalError(id) { document.getElementById(id).textContent = ''; }

function openAddPet() {
  document.getElementById('petName').value = ''; document.getElementById('petDescription').value = ''; document.getElementById('petSince').value = ''; document.getElementById('petUntil').value = '';
  document.getElementById('addPetModal').classList.add('open');
  setTimeout(() => document.getElementById('petName').focus(), 100);
}
function closeModal() { document.getElementById('addPetModal').classList.remove('open'); clearModalError('addPetError'); }

async function submitAddPet() {
  clearModalError('addPetError');
  const name = document.getElementById('petName').value.trim();
  if (!name) { modalError('addPetError', 'Name cannot be empty'); return; }
  const description = document.getElementById('petDescription').value.trim();
  if (!description) { modalError('addPetError', 'Description is required'); return; }
  const sinceRaw = document.getElementById('petSince').value;
  const untilRaw = document.getElementById('petUntil').value;
  const dateRe = /^\d{4}-\d{2}-\d{2}$/;
  if (sinceRaw && !dateRe.test(sinceRaw)) { modalError('addPetError', 'Invalid "since" date'); return; }
  if (untilRaw && !dateRe.test(untilRaw)) { modalError('addPetError', 'Invalid "until" date'); return; }
  if (sinceRaw && untilRaw && sinceRaw > untilRaw) { modalError('addPetError', '"Since" must be before "until"'); return; }
  try {
    await api('/api/pets', { method: 'POST', body: { name, description, since: sinceRaw || null, until: untilRaw || null } });
    closeModal();
    await loadPets();
    await selectPet(name);
    toast(`Created ${name}`, 'success');
  } catch(e) { toast('Error: ' + e.message, 'error'); }
}

let _petToEdit = null;

function openEditPet(name) {
  _petToEdit = name;
  const p = pets.find(p => p.name === name);
  document.getElementById('editPetName').value = p.name;
  document.getElementById('editPetDescription').value = p.description || '';
  document.getElementById('editPetSince').value = p.since || '';
  document.getElementById('editPetUntil').value = p.until || '';
  document.getElementById('editPetModal').classList.add('open');
  setTimeout(() => document.getElementById('editPetName').focus(), 100);
}
function closeEditModal() { document.getElementById('editPetModal').classList.remove('open'); _petToEdit = null; }

async function submitEditPet() {
  if (!_petToEdit) return;
  const prevActiveName = activePet?.name;
  clearModalError('editPetError');
  const name = document.getElementById('editPetName').value.trim();
  if (!name) { modalError('editPetError', 'Name cannot be empty'); return; }
  const description = document.getElementById('editPetDescription').value.trim();
  if (!description) { modalError('editPetError', 'Description is required'); return; }
  const sinceRaw = document.getElementById('editPetSince').value;
  const untilRaw = document.getElementById('editPetUntil').value;
  const dateRe = /^\d{4}-\d{2}-\d{2}$/;
  if (sinceRaw && !dateRe.test(sinceRaw)) { modalError('editPetError', 'Invalid "since" date'); return; }
  if (untilRaw && !dateRe.test(untilRaw)) { modalError('editPetError', 'Invalid "until" date'); return; }
  if (sinceRaw && untilRaw && sinceRaw > untilRaw) { modalError('editPetError', '"Since" must be before "until"'); return; }
  try {
    await api(`/api/pets/${encodeURIComponent(_petToEdit)}`, { method: 'PATCH', body: { name, description, since: sinceRaw || null, until: untilRaw || null } });
    const editedName = _petToEdit;
    closeEditModal();
    if (prevActiveName) {
      // Reselect the active pet (under its new name if it was the one edited).
      setSelection(null); resetResults();
      await loadPets();
      await selectPet(prevActiveName === editedName ? name : prevActiveName);
    } else {
      // A list or the guide is open: keep it, just refresh the sidebar.
      await loadPets();
    }
    toast('Saved', 'success');
  } catch(e) { toast('Error: ' + e.message, 'error'); }
}

let _petToDelete = null;

function openDeletePet(name) {
  _petToDelete = name;
  document.getElementById('deleteWarningText').textContent =
    `"Delete from Immich too" removes the person and untags all tagged photos in Immich permanently.`;
  document.getElementById('deleteLocalOnlyText').textContent =
    `"Remove from Pet Tagger only" keeps ${name} in Immich with all tagged photos intact, but stops auto-tagging new photos. You can re-import it later.`;
  document.getElementById('resetImmichText').textContent =
    `"Untag all photos in Immich" removes all tags for ${name} in Immich and creates a fresh person, but keeps your reference images so you can start tagging again right away.`;
  document.getElementById('deletePetModal').classList.add('open');
}
function closeDeleteModal() { document.getElementById('deletePetModal').classList.remove('open'); _petToDelete = null; }

async function confirmDeletePet(localOnly) {
  if (!_petToDelete) return;
  const name = _petToDelete;
  closeDeleteModal();
  try {
    const url = `/api/pets/${encodeURIComponent(name)}` + (localOnly ? '?local_only=true' : '');
    await api(url, { method: 'DELETE' });
    if (activePet?.name === name) {
      setSelection(null);
      showGuide();
    }
    await refreshState();
    toast(localOnly ? `Removed ${name} from Pet Tagger` : `Deleted ${name}. Immich will clean up faces in the background.`, 'success');
  } catch(e) { toast('Error: ' + e.message, 'error'); }
}

async function confirmResetPet() {
  if (!_petToDelete) return;
  const name = _petToDelete;
  closeDeleteModal();
  try {
    await api(`/api/pets/${encodeURIComponent(name)}/reset-immich`, { method: 'POST' });
    await refreshState();
    toast(`Reset ${name}: all Immich tags cleared, reference images preserved.`, 'success');
  } catch(e) { toast('Error: ' + e.message, 'error'); }
}

// ---------------------------------------------------------------------------
// Import from Immich
// ---------------------------------------------------------------------------

let _allImportPeople = [], _importSelectedPerson = null;

async function openImportPet() {
  _importSelectedPerson = null;
  _allImportPeople = [];
  document.getElementById('importSearch').value = '';
  document.getElementById('importPeopleGrid').innerHTML = '<div class="loading">Loading…</div>';
  clearModalError('importPickerError');
  document.getElementById('importPickerModal').classList.add('open');
  try {
    const d = await api('/api/immich-people');
    _allImportPeople = d.people || [];
    renderImportPeople(_allImportPeople);
  } catch(e) {
    document.getElementById('importPeopleGrid').innerHTML = `<div class="empty" style="grid-column:1/-1;padding:24px;"><div class="empty-sub">${e.message}</div></div>`;
  }
}

function renderImportPeople(people) {
  const grid = document.getElementById('importPeopleGrid');
  if (!people.length) {
    grid.innerHTML = '<div class="empty" style="grid-column:1/-1;padding:24px;"><div class="empty-sub">No people found in Immich</div></div>';
    return;
  }
  const petPersonIds = new Set(pets.map(p => p.person_id).filter(Boolean));
  grid.innerHTML = people.map(p => `
    <div class="person-card${petPersonIds.has(p.id) ? ' already-added' : ''}" data-pid="${p.id}" onclick="handlePersonCardClick(this)">
      <img class="person-thumb" src="/api/person-thumb/${p.id}" onerror="this.style.opacity=0.2" loading="lazy" alt="">
      <span class="person-name-label">${escapeHtml(p.name || '(unnamed)')}</span>
    </div>`).join('');
}

function filterImportPeople() {
  const q = document.getElementById('importSearch').value.toLowerCase();
  renderImportPeople(q ? _allImportPeople.filter(p => (p.name || '').toLowerCase().includes(q)) : _allImportPeople);
}

function handlePersonCardClick(el) {
  const id = el.dataset.pid;
  const person = _allImportPeople.find(p => p.id === id);
  if (!person) return;
  _importSelectedPerson = person;
  document.getElementById('importPickerModal').classList.remove('open');
  document.getElementById('importPetName').value = person.name || '';
  document.getElementById('importPetDescription').value = '';
  document.getElementById('importPetSince').value = '';
  document.getElementById('importPetUntil').value = '';
  clearModalError('importDetailError');
  document.getElementById('importDetailModal').classList.add('open');
  setTimeout(() => document.getElementById('importPetDescription').focus(), 100);
}

function closeImportPicker() { document.getElementById('importPickerModal').classList.remove('open'); }
function closeImportDetail() { document.getElementById('importDetailModal').classList.remove('open'); _importSelectedPerson = null; }
function backToImportPicker() { document.getElementById('importDetailModal').classList.remove('open'); document.getElementById('importPickerModal').classList.add('open'); }

async function submitImportPet() {
  const btn = document.getElementById('importPetBtn');
  if (!_importSelectedPerson || btn.disabled) return;
  clearModalError('importDetailError');
  const description = document.getElementById('importPetDescription').value.trim();
  if (!description) { modalError('importDetailError', 'Description is required'); return; }
  const sinceRaw = document.getElementById('importPetSince').value;
  const untilRaw = document.getElementById('importPetUntil').value;
  const dateRe = /^\d{4}-\d{2}-\d{2}$/;
  if (sinceRaw && !dateRe.test(sinceRaw)) { modalError('importDetailError', 'Invalid "since" date'); return; }
  if (untilRaw && !dateRe.test(untilRaw)) { modalError('importDetailError', 'Invalid "until" date'); return; }
  if (sinceRaw && untilRaw && sinceRaw > untilRaw) { modalError('importDetailError', '"Since" must be before "until"'); return; }
  // Import fetches faces and runs YOLO on up to 200 photos, so it can take minutes.
  btn.disabled = true;
  btn.textContent = 'Importing…';
  try {
    const result = await api('/api/pets/import', { method: 'POST', body: {
      person_id: _importSelectedPerson.id,
      name: _importSelectedPerson.name,
      description,
      since: sinceRaw || null,
      until: untilRaw || null,
    }});
    closeImportDetail();
    await refreshState();
    await selectPet(result.name);
    toast(result.ref_count > 0 ? `Imported ${result.name} with ${result.ref_count} refs` : `Imported ${result.name} with 0 refs. No animals were detected in the reference photos. Add refs manually.`, result.ref_count > 0 ? 'success' : 'warn');
  } catch(e) { modalError('importDetailError', e.message); }
  finally {
    btn.disabled = false;
    btn.textContent = 'Import';
  }
}

// ---------------------------------------------------------------------------
// Modal backdrop dismissal
// ---------------------------------------------------------------------------

document.getElementById('addPetModal').addEventListener('click', function(e) { if (e.target === this) closeModal(); });
document.getElementById('editPetModal').addEventListener('click', function(e) { if (e.target === this) closeEditModal(); });
document.getElementById('deletePetModal').addEventListener('click', function(e) { if (e.target === this) closeDeleteModal(); });
document.getElementById('importPickerModal').addEventListener('click', function(e) { if (e.target === this) closeImportPicker(); });
document.getElementById('importDetailModal').addEventListener('click', function(e) { if (e.target === this) closeImportDetail(); });

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

(async () => {
  const savedTileSize = parseInt(localStorage.getItem(TILE_SIZE_KEY), 10);
  if (savedTileSize) document.documentElement.style.setProperty('--tile-size', savedTileSize + 'px');
  await refreshState();
  if (!activePet && !activeList && pets.length > 0) showGuide();
  prefillScanUntil();
  loadScanResult();
  api('/api/version').then(async d => {
    const el = document.getElementById('versionLabel');
    if (!el) return;
    const current = d.version;
    el.textContent = current;
    try {
      const CACHE_KEY = 'pet_tagger_latest_version';
      const CACHE_TTL = 3600 * 1000;
      const cached = JSON.parse(localStorage.getItem(CACHE_KEY) || 'null');
      let latest = cached && (Date.now() - cached.ts < CACHE_TTL) ? cached.version : null;
      if (!latest) {
        const r = await fetch('https://api.github.com/repos/tedornitier/immich-pet-tagger/releases/latest');
        if (r.ok) {
          latest = (await r.json()).tag_name;
          localStorage.setItem(CACHE_KEY, JSON.stringify({ version: latest, ts: Date.now() }));
        }
      }
      // Only when the release is newer: a cached older tag right after updating, or a
      // dev build ahead of the last release, must not show "update".
      const parts = v => String(v).replace(/^v/, '').split('.').map(n => parseInt(n, 10) || 0);
      const newer = (a, b) => { const x = parts(a), y = parts(b); for (let i = 0; i < 3; i++) if ((x[i] || 0) !== (y[i] || 0)) return (x[i] || 0) > (y[i] || 0); return false; };
      if (latest && newer(latest, current)) {
        el.innerHTML = `${current} <a href="https://github.com/tedornitier/immich-pet-tagger/releases/latest" target="_blank" rel="noopener" style="color:var(--accent);text-decoration:none;font-weight:600;" title="Update available: ${latest}">↑ update</a>`;
      }
    } catch(_) {}
  }).catch(() => {});
})();
