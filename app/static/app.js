let pets = [], activePet = null, selectedCrops = new Map(), refsItems = [], negIds = [], immichUrl = 'http://localhost:2283', negCandidateMode = false, borderlineMode = false, scanLowConfMode = false, lastClickedKey = null, negGeneration = 0, negPollTimer = null, blGeneration = 0, blPollTimer = null;

async function api(path, opts = {}) {
  const r = await fetch(path, { headers: { 'Content-Type': 'application/json' }, ...opts, body: opts.body ? JSON.stringify(opts.body) : undefined });
  if (!r.ok) { const t = await r.text().catch(() => r.statusText); throw new Error(t); }
  return r.json();
}

function toast(msg, type = '') {
  const el = document.getElementById('toast');
  el.textContent = msg; el.className = 'toast show' + (type ? ' ' + type : '');
  clearTimeout(el._t); el._t = setTimeout(() => el.className = 'toast', 4000);
}

function initials(name) { return name.slice(0, 2).toUpperCase(); }

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
}

// ---------------------------------------------------------------------------
// Pets
// ---------------------------------------------------------------------------

async function loadPets(keepActive = false) {
  try {
    const d = await api('/api/pets');
    pets = d.pets;
    if (activePet) {
      activePet = pets.find(p => p.name === activePet.name) || activePet;
    }
    renderSidebar();
    updateNegStatus();
  } catch(e) { toast('Could not load pets: ' + e.message, 'error'); }
}

function renderSidebar() {
  const el = document.getElementById('petsList');
  if (!pets.length) {
    el.innerHTML = '<div style="padding:16px;font-size:12px;color:var(--text3);text-align:center;line-height:1.6;">No pets yet.<br>Add one to get started.</div>';
    showGuide();
    document.getElementById('refsTitle').textContent = 'No pet selected';
    document.getElementById('findRefsBtn').style.display = 'none';
    document.getElementById('addByIdBtn').style.display = 'none';
    document.getElementById('clearRefsBtn').style.display = 'none';
    document.getElementById('refsGrid').innerHTML = '<div class="empty" style="grid-column:1/-1;height:200px;"><div class="empty-sub">Add a pet first</div></div>';
    return;
  }
  el.innerHTML = pets.map(p => `
    <div class="pet-item ${activePet?.name === p.name ? 'active' : ''}" onclick="selectPet('${p.name}')">
      <div class="pet-avatar">${p.person_id ? `<img src="/api/person-thumb/${p.person_id}" onerror="this.parentElement.textContent='${initials(p.name)}'" alt="">` : initials(p.name)}</div>
      <div class="pet-info">
        <div class="pet-name">${p.name}</div>
        <div class="pet-count">${p.ref_count} ref${p.ref_count !== 1 ? 's' : ''}</div>
      </div>
      <button class="pet-edit" onclick="event.stopPropagation(); openEditPet('${p.name}')" title="Edit">✎</button>
      <button class="pet-delete" onclick="event.stopPropagation(); openDeletePet('${p.name}')" title="Delete">✕</button>
    </div>`).join('');
}

function showGuide() {
  document.getElementById('resultsLabel').textContent = '';
  document.getElementById('photoGrid').innerHTML = `<div class="guide" style="grid-column:1/-1">
    <div class="guide-steps">
      <div class="guide-step"><div class="guide-step-num">1</div><div class="guide-step-body"><div class="guide-step-title">Add your pet</div><div class="guide-step-desc">Click <strong>↓ Import from Immich</strong> if Immich already recognizes your pet as a person. Otherwise click <strong>+ Add pet</strong> to start from scratch.</div></div></div>
      <div class="guide-step"><div class="guide-step-num">2</div><div class="guide-step-body"><div class="guide-step-title">Find reference photos</div><div class="guide-step-desc">Select your pet and click <strong>Find references</strong>. Aim for 20–30 to start; results improve up to around 50.<ul style="margin:6px 0 0 16px;padding:0;"><li><strong>Add to pet</strong>: clear, close-up shot, your pet is the only subject.</li><li><strong>Ignore</strong>: blurry, distant, another person or animal visible alongside your pet, or a look-alike that is not yours. Ignored photos won't appear again.</li><li><strong>Not a pet</strong>: photos that could confuse the classifier. Empty rooms, other species, ambiguous shots. Around 50 is enough.</li></ul>If you already know a specific photo you want to use, click <strong>Add manually</strong> and paste its Immich URL or asset ID.</div></div></div>
      <div class="guide-step"><div class="guide-step-num">3</div><div class="guide-step-body"><div class="guide-step-title">Add "not a pet" samples</div><div class="guide-step-desc">These teach the classifier what not to tag: empty rooms, other animals of a different species, ambiguous shots with no clear subject. Without them, the classifier will tag almost anything. In the <strong>Not a pet</strong> panel, click <strong>Find candidates</strong> to automatically surface more photos that might confuse the classifier. To add a specific photo directly, click <strong>Add manually</strong> and paste its Immich URL or asset ID.</div></div></div>
      <div class="guide-step"><div class="guide-step-num">4</div><div class="guide-step-body"><div class="guide-step-title">Run a test scan</div><div class="guide-step-desc">Set the <strong>Scan from</strong> date 1–2 weeks back and click <strong>Scan</strong>. Review low confidence results: add correct ones as refs, and click <strong>Ignore</strong> on the rest. Ignored photos won't appear again.</div></div></div>
      <div class="guide-step"><div class="guide-step-num">5</div><div class="guide-step-body"><div class="guide-step-title">Iterate</div><div class="guide-step-desc">Repeat steps 2–4 a couple of times. Results typically stabilize after 2–3 rounds.</div></div></div>
      <div class="guide-step"><div class="guide-step-num">6</div><div class="guide-step-body"><div class="guide-step-title">Run the full backfill</div><div class="guide-step-desc">Once happy with accuracy, set the scan date to when you got your pet and run the full scan. After that, new photos are tagged automatically every hour.</div></div></div>
    </div>
  </div>`;
  selectedCrops.clear(); lastClickedKey = null; updateSelUI();
}

function clearSearch() {
  document.getElementById('resultsLabel').textContent = '';
  document.getElementById('photoGrid').innerHTML = '<div class="empty" style="grid-column:1/-1; height:300px;"><div class="empty-icon">🐾</div><div class="empty-title">Find photos</div><div class="empty-sub">Click "Find references" to get started</div></div>';
  selectedCrops.clear(); lastClickedKey = null; updateSelUI();
}

async function selectPet(name) {
  if (activePet?.name === name) return;
  if (selectedCrops.size > 0) {
    const ok = confirm(`You have ${selectedCrops.size} selected photo${selectedCrops.size !== 1 ? 's' : ''} not yet assigned. Switch anyway?`);
    if (!ok) return;
  }
  negCandidateMode = false; borderlineMode = false; scanLowConfMode = false;
  activePet = pets.find(p => p.name === name);
  clearSearch(); renderSidebar();
  document.getElementById('refsTitle').textContent = name;
  document.getElementById('findRefsBtn').style.display = '';
  document.getElementById('addByIdBtn').style.display = '';
  document.getElementById('clearRefsBtn').style.display = '';
  await loadRefs(name);
  await loadNegatives();
}

async function loadRefs(name) {
  const grid = document.getElementById('refsGrid');
  grid.innerHTML = '<div class="loading">Loading…</div>';
  try {
    const d = await api(`/api/pets/${encodeURIComponent(name)}/assets`);
    refsItems = d.assets;
    renderRefs(d.assets);
  } catch(e) { grid.innerHTML = '<div class="empty" style="grid-column:1/-1"><div class="empty-sub">Error loading refs</div></div>'; }
}

function renderRefs(assets) {
  const grid = document.getElementById('refsGrid');
  if (!assets.length) { grid.innerHTML = '<div class="empty" style="grid-column:1/-1;height:160px;"><div class="empty-sub">No references yet.<br>Click "Find references" to add some.</div></div>'; return; }
  grid.innerHTML = assets.map(a => {
    const cropArg = a.crop_idx != null ? a.crop_idx : 'null';
    return `<div class="ref-thumb">
      <a href="${immichUrl}/photos/${a.id}" target="_blank" rel="noopener" title="Open in Immich">
        <img src="${a.thumb}" loading="lazy" onerror="this.style.opacity=0.2">
      </a>
      <button class="ref-remove" onclick="removeRef('${a.id}', ${cropArg})" title="Remove">✕</button>
    </div>`;
  }).join('');
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
    const grid = document.getElementById('refsGrid');
    const scrollTop = grid.scrollTop;
    renderRefs(refsItems);
    grid.scrollTop = scrollTop;
    await refreshState();
    toast('Removed');
  } catch(e) { toast('Error: ' + e.message, 'error'); }
}

async function assignSelected() {
  if (!activePet || !selectedCrops.size) return;
  const pet = activePet;
  const newCrops = [...selectedCrops.values()];
  const existing = refsItems.map(r => ({ asset_id: r.id, crop_idx: r.crop_idx, bbox: r.bbox }));
  const seen = new Set();
  const merged = [...existing, ...newCrops].filter(c => {
    const k = c.crop_idx != null ? `${c.asset_id}_${c.crop_idx}` : c.asset_id;
    if (seen.has(k)) return false;
    seen.add(k); return true;
  });
  try {
    await api(`/api/pets/${encodeURIComponent(pet.name)}/assets`, { method: 'POST', body: { assets: merged } });
    selectedCrops.clear(); updateSelUI();
    document.querySelectorAll('.photo-thumb.selected').forEach(el => { el.classList.remove('selected'); el.classList.add('is-ref'); });
    await loadRefs(pet.name);
    await refreshState();
    toast(`Added to ${pet.name}`, 'success');
  } catch(e) { toast('Error: ' + e.message, 'error'); }
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
  // No animal was detected, so this is the whole photo and the score describes
  // the whole scene. Flag it: accepting one as a reference teaches the
  // classifier to match on background instead of on the pet.
  const fullImageBadge = a.full_image ? '<div class="full-image-badge">Full photo</div>' : '';
  const title = a.pet_name
    ? `${fmtDate(a.date)} · ${Math.round(a.score * 100)}% ${a.pet_name}` +
      (a.full_image ? ' · whole photo, no animal detected' : '')
    : `${a.filename || ''} · ${fmtDate(a.date)}`;
  const makeItem = (key, src, cropIdx, bbox) => {
    const cropIdxAttr = cropIdx != null ? `data-crop-idx="${cropIdx}"` : '';
    const bboxAttr = bbox ? `data-bbox='${JSON.stringify(bbox)}'` : '';
    return `<div class="photo-thumb" id="th-${key}" data-asset-id="${a.id}" ${cropIdxAttr} ${bboxAttr}
      onclick="toggleSelect(event,'${key}')" title="${title}">
      <img src="${src}" loading="lazy" onerror="this.src='data:image/svg+xml,<svg/>'">
      <a class="photo-open" href="${immichUrl}/photos/${a.id}" target="_blank" rel="noopener" onclick="event.stopPropagation()">⤢</a>
      <div class="photo-check">✓</div>
      ${badge}
      ${fullImageBadge}
    </div>`;
  };
  if (a.crops && a.crops.length > 0) {
    return a.crops.map(c => makeItem(`${a.id}_${c.crop_idx}`, `/api/crop/${a.id}?bbox=${c.bbox.join(',')}`, c.crop_idx, c.bbox));
  }
  return [makeItem(a.id, a.thumb, null, a.bbox || null)];
}

function markGridItems(assets) {
  const refKeys = new Set(refsItems.map(r => r.crop_idx != null ? `${r.id}_${r.crop_idx}` : r.id));
  const negSet = new Set(negIds);
  assets.forEach(a => {
    const keys = (a.crops && a.crops.length > 0) ? a.crops.map(c => `${a.id}_${c.crop_idx}`) : [a.id];
    keys.forEach(key => {
      const el = document.getElementById('th-' + key);
      if (!el) return;
      if (refKeys.has(key) || refKeys.has(a.id)) el.classList.add('is-ref');
      if (negSet.has(a.id)) el.classList.add('is-neg');
    });
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
  clearModalError('addByIdError');
  const example = 'e.g. a1b2c3d4-11aa-22bb-33cc-4d5e6f7a8b9c';
  if (mode === 'neg') {
    document.getElementById('addByIdTitle').textContent = 'Add "not a pet" by ID or link';
    document.getElementById('addByIdHint').textContent = `Paste an Immich photo URL or just the asset ID (${example}). It will be added directly to "not a pet".`;
  } else {
    document.getElementById('addByIdTitle').textContent = 'Add reference by ID or link';
    document.getElementById('addByIdHint').textContent = `Paste an Immich photo URL or just the asset ID (${example}). If one animal is detected it is added immediately; multiple crops let you pick the right one.`;
  }
  document.getElementById('addByIdModal').classList.add('open');
  setTimeout(() => document.getElementById('addByIdInput').focus(), 100);
}

function openAddById() { if (activePet) _openAddByIdModal('ref'); }
function openAddNegById() { _openAddByIdModal('neg'); }

function closeAddById() {
  document.getElementById('addByIdModal').classList.remove('open');
  clearModalError('addByIdError');
}

async function submitAddById() {
  clearModalError('addByIdError');
  const id = parseAssetId(document.getElementById('addByIdInput').value);
  if (!id) { modalError('addByIdError', 'Could not find an asset ID. Paste an Immich photo link or the bare ID.'); return; }

  if (_addByIdMode === 'neg') {
    try {
      await api(`/api/asset/${id}/crops`); // validates asset exists
      await api('/api/negatives', { method: 'POST', body: { asset_ids: [id] } });
      closeAddById();
      await loadNegatives();
      toast('Added to "not a pet"', 'success');
    } catch(e) {
      let msg = e.message || 'Could not load asset';
      try { const j = JSON.parse(msg); if (j.detail) msg = j.detail; } catch(_) {}
      modalError('addByIdError', msg);
    }
    return;
  }

  // ref mode
  if (!activePet) return;
  const pet = activePet;
  try {
    const a = await api(`/api/asset/${id}/crops`);
    const crops = a.crops || [];
    if (crops.length <= 1) {
      // 0 or 1 crop: add immediately, no picker needed
      const existing = refsItems.map(r => ({ asset_id: r.id, crop_idx: r.crop_idx, bbox: r.bbox }));
      const newCrop = crops.length === 1
        ? { asset_id: id, crop_idx: crops[0].crop_idx, bbox: crops[0].bbox }
        : { asset_id: id, crop_idx: null, bbox: null };
      const alreadyPresent = existing.some(r => r.asset_id === id);
      if (!alreadyPresent) existing.push(newCrop);
      await api(`/api/pets/${encodeURIComponent(pet.name)}/assets`, { method: 'POST', body: { assets: existing } });
      closeAddById();
      await loadRefs(pet.name);
      await refreshState();
      toast(`Added to ${pet.name}`, 'success');
    } else {
      // Multiple crops: load into grid so user picks the right animal
      closeAddById();
      negCandidateMode = false; borderlineMode = false; scanLowConfMode = false;
      selectedCrops.clear(); lastClickedKey = null; updateSelUI();
      document.getElementById('resultsLabel').textContent = `${crops.length} pets detected. Select the one to add as reference`;
      document.getElementById('photoGrid').innerHTML = renderPhotoItems(a, 0.8).join('');
      markGridItems([a]);
    }
  } catch(e) {
    let msg = e.message || 'Could not load asset';
    try { const j = JSON.parse(msg); if (j.detail) msg = j.detail; } catch(_) {}
    modalError('addByIdError', msg);
  }
}

async function viewSuggestions() {
  if (!activePet) return;
  const pet = activePet;
  if (!pet.description) { toast('Edit this pet and add a description to use this feature', 'error'); return; }
  selectedCrops.clear(); lastClickedKey = null; updateSelUI();
  const grid = document.getElementById('photoGrid');
  const label = document.getElementById('resultsLabel');
  grid.innerHTML = '<div class="loading" style="grid-column:1/-1">Finding similar photos… this may take a moment</div>';
  label.textContent = 'Finding references…';
  try {
    const d = await api(`/api/pets/${encodeURIComponent(pet.name)}/suggestions`);
    label.textContent = `${d.assets.length} photo${d.assets.length !== 1 ? 's' : ''} similar to ${pet.name}'s refs`;
    if (!d.assets.length) {
      grid.innerHTML = '<div class="empty" style="grid-column:1/-1;height:200px;"><div class="empty-icon">🐾</div><div class="empty-title">No suggestions found</div><div class="empty-sub">Add more refs or broaden the date range</div></div>';
      return;
    }
    grid.innerHTML = d.assets.flatMap(a => renderPhotoItems(a, 0.8)).join('');
    markGridItems(d.assets);
  } catch(e) {
    label.textContent = 'Failed to load suggestions';
    grid.innerHTML = `<div class="empty" style="grid-column:1/-1;height:200px;"><div class="empty-sub">${e.message}</div></div>`;
    toast('Suggestions error: ' + e.message, 'error');
  }
}

async function viewBorderline() {
  if (!activePet || !activePet.ref_count) return;
  const myGen = ++blGeneration;
  if (blPollTimer) { clearInterval(blPollTimer); blPollTimer = null; }
  negCandidateMode = false; borderlineMode = true;
  selectedCrops.clear(); lastClickedKey = null; updateSelUI();
  const grid = document.getElementById('photoGrid');
  const label = document.getElementById('resultsLabel');
  const petName = activePet.name;
  grid.innerHTML = '<div class="loading" id="blLoadMsg" style="grid-column:1/-1">Loading…</div>';
  label.textContent = 'Finding references…';

  blPollTimer = setInterval(async () => {
    if (blGeneration !== myGen) { clearInterval(blPollTimer); blPollTimer = null; return; }
    try {
      const p = await api(`/api/pets/${encodeURIComponent(petName)}/borderline/progress`);
      const el = document.getElementById('blLoadMsg');
      if (!el) return;
      if (p.total > 0) el.textContent = `Loading ${Math.round(p.current / p.total * 100)}%…`;
      else if (p.running) el.textContent = 'Loading…';
    } catch(_) {}
  }, 1000);

  try {
    const d = await api(`/api/pets/${encodeURIComponent(petName)}/borderline`);
    clearInterval(blPollTimer); blPollTimer = null;
    if (blGeneration !== myGen) return;
    label.textContent = `${d.assets.length} photo${d.assets.length !== 1 ? 's' : ''} ${petName} might be missing. Add good ones as refs to improve accuracy.`;
    if (!d.assets.length) {
      grid.innerHTML = '<div class="empty" style="grid-column:1/-1;height:200px;"><div class="empty-icon">🐾</div><div class="empty-title">No missed photos found</div><div class="empty-sub">The classifier is either very confident or not finding this pet at all</div></div>';
      return;
    }
    const thr = d.threshold ?? 0.8;
    grid.innerHTML = d.assets.flatMap(a => renderPhotoItems(a, thr)).join('');
    markGridItems(d.assets);
  } catch(e) {
    clearInterval(blPollTimer); blPollTimer = null;
    if (blGeneration !== myGen) return;
    label.textContent = 'Failed to load missed photos';
    grid.innerHTML = `<div class="empty" style="grid-column:1/-1;height:200px;"><div class="empty-sub">${e.message}</div></div>`;
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
  document.getElementById('assignBtn').style.display = (n && activePet && !negCandidateMode && !scanLowConfMode) ? '' : 'none';
  document.getElementById('skipBtn').style.display = n ? '' : 'none';
  document.getElementById('addNegBtn').style.display = n ? '' : 'none';
  document.getElementById('scanPetBtns').style.display = (n && scanLowConfMode) ? 'flex' : 'none';
}

async function skipSelected() {
  if (!selectedCrops.size) return;
  const ids = [...new Set([...selectedCrops.values()].map(c => c.asset_id))];
  try {
    await api('/api/skipped', { method: 'POST', body: { asset_ids: ids } });
    ids.forEach(id => document.querySelectorAll(`[data-asset-id="${id}"]`).forEach(el => el.remove()));
    selectedCrops.clear(); updateSelUI();
    toast(`Ignored ${ids.length} photo${ids.length !== 1 ? 's' : ''}. Won't appear again.`, 'success');
  } catch(e) { toast('Error: ' + e.message, 'error'); }
}

// ---------------------------------------------------------------------------
// Negatives
// ---------------------------------------------------------------------------

function updateNegStatus() {
  const el = document.getElementById('negCount');
  el.textContent = negIds.length;
  el.style.color = '';
}

async function loadNegatives() {
  try {
    const d = await api('/api/negatives');
    negIds = d.assets.map(a => a.id);
    updateNegStatus();
    document.getElementById('clearNegsBtn').style.display = negIds.length ? '' : 'none';
    const grid = document.getElementById('negGrid');
    if (!negIds.length) { grid.innerHTML = ''; return; }
    grid.innerHTML = d.assets.map(a => `
      <div class="ref-thumb">
        <a href="${immichUrl}/photos/${a.id}" target="_blank" rel="noopener" title="Open in Immich">
          <img src="${a.thumb}" loading="lazy" onerror="this.style.opacity=0.2">
        </a>
        <button class="ref-remove" onclick="removeNegative('${a.id}')" title="Remove">✕</button>
      </div>`).join('');
  } catch(e) { console.warn('loadNegatives:', e); }
}

async function addSelectedAsNegatives() {
  if (!selectedCrops.size) return;
  const assetIds = [...new Set([...selectedCrops.values()].map(c => c.asset_id))];
  try {
    await api('/api/negatives', { method: 'POST', body: { asset_ids: assetIds } });
    negIds = [...new Set([...negIds, ...assetIds])];
    selectedCrops.forEach((_, key) => {
      const el = document.getElementById('th-' + key);
      if (el) { el.classList.remove('selected'); el.classList.add('is-neg'); }
    });
    selectedCrops.clear(); updateSelUI();
    await loadNegatives();
    toast('Added to "not my pets"', 'success');
  } catch(e) { toast('Error: ' + e.message, 'error'); }
}

async function viewNegCandidates() {
  const myGen = ++negGeneration;
  if (negPollTimer) { clearInterval(negPollTimer); negPollTimer = null; }
  negCandidateMode = true;
  selectedCrops.clear(); lastClickedKey = null; updateSelUI();
  const grid = document.getElementById('photoGrid');
  const label = document.getElementById('resultsLabel');
  grid.innerHTML = '<div class="loading" id="negLoadMsg" style="grid-column:1/-1">Loading…</div>';
  label.textContent = 'Finding candidates…';

  negPollTimer = setInterval(async () => {
    if (negGeneration !== myGen) { clearInterval(negPollTimer); negPollTimer = null; return; }
    try {
      const p = await api('/api/suggestions/negatives/progress');
      const el = document.getElementById('negLoadMsg');
      if (!el) return;
      if (p.total > 0) el.textContent = `Loading ${Math.round(p.current / p.total * 100)}%…`;
      else if (p.running) el.textContent = 'Loading…';
    } catch(_) {}
  }, 1000);

  try {
    const d = await api('/api/suggestions/negatives');
    clearInterval(negPollTimer); negPollTimer = null;
    if (negGeneration !== myGen) return;
    updateNegStatus();
    label.textContent = `${d.assets.length} candidate${d.assets.length !== 1 ? 's' : ''} for "not my pets"`;
    if (!d.assets.length) {
      grid.innerHTML = '<div class="empty" style="grid-column:1/-1;height:200px;"><div class="empty-icon">🐾</div><div class="empty-title">No candidates found</div><div class="empty-sub">Classifier is well calibrated</div></div>';
      return;
    }
    const thr = d.threshold || 0.8;
    grid.innerHTML = d.assets.flatMap(a => renderPhotoItems(a, thr)).join('');
    const negSet = new Set(negIds);
    d.assets.forEach(a => {
      if (negSet.has(a.id)) document.getElementById('th-' + a.id)?.classList.add('is-neg');
    });
  } catch(e) {
    clearInterval(negPollTimer); negPollTimer = null;
    if (negGeneration !== myGen) return;
    label.textContent = 'Failed to load candidates';
    grid.innerHTML = `<div class="empty" style="grid-column:1/-1;height:200px;"><div class="empty-sub">${e.message}</div></div>`;
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
  if (!confirm(`Remove all "not my pets" photos from Pet Tagger? This will not affect Immich.`)) return;
  try {
    await api('/api/negatives/all', { method: 'DELETE' });
    negIds = [];
    await loadNegatives();
    toast('All "not my pets" cleared', 'success');
  } catch(e) { toast('Error: ' + e.message, 'error'); }
}

async function removeNegative(id) {
  try {
    await api(`/api/negatives/${id}`, { method: 'DELETE' });
    negIds = negIds.filter(i => i !== id);
    await loadNegatives();
    document.getElementById('th-' + id)?.classList.remove('is-neg');
    toast('Removed from "not my pets"');
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


function relativeTime(iso) {
  const diff = Math.floor((Date.now() - new Date(iso)) / 1000);
  if (diff < 60) return `${diff}s ago`;
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return `${Math.floor(diff / 86400)}d ago`;
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
      stat('Full photo', c.full_image || 0, 'nonzero-warn') +
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
    // Both buckets land in the same review queue: below-threshold matches and
    // whole-photo matches, which are never auto-tagged however high they score.
    const reviewCount = (c.low_confidence || 0) + (c.full_image || 0);
    el.innerHTML = '<div class="scan-result-header">Scan result</div>' +
      '<div class="poll-stats" style="margin-top:6px;">' +
      stat('Tagged', c.added, 'nonzero-good') +
      stat('Low conf.', c.low_confidence, 'nonzero-warn') +
      stat('Full photo', c.full_image || 0, 'nonzero-warn') +
      stat('Other', c.unknown, '') +
      stat('Out of range', c.out_of_range, '') +
      stat('Already tagged', c.already_tagged, '') +
      (c.failed > 0 ? stat('Failed', c.failed, 'nonzero-bad') : '') +
      (c.no_thumb > 0 ? stat('No thumb', c.no_thumb, 'nonzero-warn') : '') +
      '</div>' +
      (reviewCount > 0 ? `<button class="btn" style="font-size:11px;margin-top:8px;width:100%;" onclick="viewScanLowConf()">Review ${reviewCount} for tagging</button>` : '');
  }
}

async function viewScanLowConf() {
  scanLowConfMode = true;
  negCandidateMode = false; borderlineMode = false;
  selectedCrops.clear(); lastClickedKey = null;
  const grid = document.getElementById('photoGrid');
  const label = document.getElementById('resultsLabel');
  grid.innerHTML = '<div class="loading" style="grid-column:1/-1">Loading results to review…</div>';
  label.textContent = 'Loading…';
  const scanPetBtns = document.getElementById('scanPetBtns');
  scanPetBtns.innerHTML = pets.map(p => `<button class="btn btn-primary" title="Clear, close-up shot, your pet is the only subject.">${p.name}</button>`).join('');
  [...scanPetBtns.children].forEach((btn, i) => { btn.onclick = () => scanAssignSelected(pets[i].name); });
  updateSelUI();
  try {
    const d = await api('/api/scan/low-confidence');
    if (!d.assets.length) {
      label.textContent = 'Nothing to review';
      grid.innerHTML = '<div class="empty" style="grid-column:1/-1; height:200px;"><div class="empty-sub">Every match was either tagged or rejected outright</div></div>';
      return;
    }
    label.textContent = `${d.assets.length} result${d.assets.length !== 1 ? 's' : ''} to review`;
    const thr = d.threshold ?? 0.8;
    const negSet = new Set(negIds);
    grid.innerHTML = d.assets.flatMap(a => renderPhotoItems(a, thr)).join('');
    d.assets.forEach(a => { if (negSet.has(a.id)) document.getElementById('th-' + a.id)?.classList.add('is-neg'); });
  } catch(e) {
    label.textContent = 'Failed to load';
    grid.innerHTML = `<div class="empty" style="grid-column:1/-1; height:200px;"><div class="empty-sub">${e.message}</div></div>`;
  }
}

async function scanAssignSelected(petName) {
  if (!selectedCrops.size) return;
  const newCrops = [...selectedCrops.values()];
  try {
    const existing = await api(`/api/pets/${encodeURIComponent(petName)}/assets`);
    const existingCrops = existing.assets.map(a => ({ asset_id: a.id, crop_idx: a.crop_idx, bbox: a.bbox }));
    const seen = new Set();
    const merged = [...existingCrops, ...newCrops].filter(c => {
      const k = c.crop_idx != null ? `${c.asset_id}_${c.crop_idx}` : c.asset_id;
      if (seen.has(k)) return false;
      seen.add(k); return true;
    });
    await api(`/api/pets/${encodeURIComponent(petName)}/assets`, { method: 'POST', body: { assets: merged } });
    selectedCrops.forEach((_, key) => {
      const el = document.getElementById('th-' + key);
      if (el) { el.classList.remove('selected'); el.classList.add('is-ref'); }
    });
    selectedCrops.clear(); updateSelUI();
    await refreshState();
    toast(`Added ${newCrops.length} to ${petName}`, 'success');
  } catch(e) { toast(e.message, 'error'); }
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
    await loadPets(true);
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
    closeEditModal();
    activePet = null; clearSearch();
    await loadPets(true);
    const selectName = prevActiveName === _petToEdit ? name : (prevActiveName || pets[0]?.name);
    if (selectName) await selectPet(selectName);
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
      activePet = null;
      document.getElementById('refsTitle').textContent = 'No pet selected';
      document.getElementById('refsGrid').innerHTML = '<div class="empty" style="grid-column:1/-1;height:200px;"><div class="empty-sub">Select a pet</div></div>';
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
      <span class="person-name-label">${p.name || '—'}</span>
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
  if (!_importSelectedPerson) return;
  clearModalError('importDetailError');
  const description = document.getElementById('importPetDescription').value.trim();
  if (!description) { modalError('importDetailError', 'Description is required'); return; }
  const sinceRaw = document.getElementById('importPetSince').value;
  const untilRaw = document.getElementById('importPetUntil').value;
  const dateRe = /^\d{4}-\d{2}-\d{2}$/;
  if (sinceRaw && !dateRe.test(sinceRaw)) { modalError('importDetailError', 'Invalid "since" date'); return; }
  if (untilRaw && !dateRe.test(untilRaw)) { modalError('importDetailError', 'Invalid "until" date'); return; }
  if (sinceRaw && untilRaw && sinceRaw > untilRaw) { modalError('importDetailError', '"Since" must be before "until"'); return; }
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
  await refreshState();
  if (!activePet && pets.length > 0) showGuide();
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
      if (latest && latest !== current) {
        el.innerHTML = `${current} <a href="https://github.com/tedornitier/immich-pet-tagger/releases/latest" target="_blank" rel="noopener" style="color:var(--accent);text-decoration:none;font-weight:600;" title="Update available: ${latest}">↑ update</a>`;
      }
    } catch(_) {}
  }).catch(() => {});
})();
