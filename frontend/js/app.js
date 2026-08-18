/* ── MERLIN Frontend ───────────────────────────────────────────────────── */
const API = 'http://localhost:5000';
let lastResult = null;
let queuedFiles = [];
const queryInput = document.getElementById('query-input');
const topKInput = document.getElementById('top-k-input');

let pendingPapers = [];
let pipelineState = { status: 'Idle', stage: 'Ready' };

function updatePipelineStatus(status, stage) {
  pipelineState = { status, stage };
  const card = document.getElementById('pipeline-status-card');
  const statusValue = document.getElementById('pipeline-status-value');
  const stageValue = document.getElementById('pipeline-stage-value');

  if (card) {
    card.classList.remove('is-idle', 'is-running', 'is-complete', 'is-error');
    const normalized = String(status || '').toLowerCase();
    if (normalized.includes('error') || normalized.includes('fail')) card.classList.add('is-error');
    else if (normalized.includes('complete') || normalized.includes('ready')) card.classList.add('is-complete');
    else if (normalized.includes('search') || normalized.includes('analyz') || normalized.includes('running') || normalized.includes('load')) card.classList.add('is-running');
    else card.classList.add('is-idle');
  }

  if (statusValue) statusValue.textContent = status || 'Idle';
  if (stageValue) stageValue.textContent = stage || 'Ready';
}

function updateRunButtonState() {
  const btn = document.getElementById('btn-fetch');
  const analyseBtn = document.getElementById('btn-analyse');
  const input = document.getElementById('query-input');
  const hasQuery = !!(input && input.value.trim());
  if (btn) btn.disabled = !(hasQuery || queuedFiles.length);
  if (analyseBtn) analyseBtn.style.display = hasQuery ? 'none' : (queuedFiles.length ? 'flex' : 'none');
  if (btn) {
    btn.style.display = hasQuery ? 'flex' : (queuedFiles.length ? 'none' : 'flex');
  }
}

window.applyPreset = function(type) {
  const presets = {
    metformin: "Metformin reduces oxidative stress and modulates aging pathways",
    nsclc: "Pembrolizumab versus chemotherapy overall survival in NSCLC",
    ad: "Anti-amyloid monoclonal antibodies versus tau neuroinflammation in Alzheimer's disease",
  };
  const q = presets[type];
  if (!q) return;
  const input = document.getElementById('query-input');
  if (input) {
    input.value = q;
    updateRunButtonState();
    input.focus();
    const btn = document.getElementById('btn-fetch');
    if (btn && !btn.disabled) btn.click();
  }
};

window.toggleFullscreenGraph = function() {
  const wrap = document.querySelector('.edg-wrap');
  const canvas = document.getElementById('edg-canvas');
  const btn = document.getElementById('btn-expand-canvas');
  if (!wrap || !canvas) return;
  wrap.classList.toggle('fullscreen-edg');
  const isFull = wrap.classList.contains('fullscreen-edg');
  if (btn) btn.textContent = isFull ? '✕ Collapse Canvas' : '⛶ Expand Canvas';
  if (isFull) {
    canvas.width = window.innerWidth - 60;
    canvas.height = window.innerHeight - 160;
  } else {
    canvas.width = 760;
    canvas.height = 420;
  }
  _edgLayoutSig = '';
  if (lastResult && lastResult.graph) renderEDG(lastResult.graph);
};

/* ── THEME ─────────────────────────────────────────────────────────────── */
(function () {
  document.documentElement.setAttribute('data-theme',
    localStorage.getItem('merlin-theme') || 'dark');
})();
document.getElementById('theme-toggle').addEventListener('click', () => {
  const html = document.documentElement;
  const next = html.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
  html.setAttribute('data-theme', next);
  localStorage.setItem('merlin-theme', next);
  if (lastResult && !document.getElementById('tab-graph').classList.contains('hidden'))
    renderEDG(lastResult.graph || {});
});

/* ── API HEALTH ────────────────────────────────────────────────────────── */
async function checkHealth() {
  const s = document.getElementById('api-status');
  try {
    const r = await fetch(`${API}/api/health`, { signal: AbortSignal.timeout(4000) });
    if (r.ok) {
      s.className = 'status-dot ok';
      s.querySelector('.status-label').textContent = 'API Online';
    } else throw 0;
  } catch {
    s.className = 'status-dot error';
    s.querySelector('.status-label').textContent = 'API Offline';
  }
}
checkHealth();
setInterval(checkHealth, 30000);

/* ── MODEL INFO ────────────────────────────────────────────────────────── */
async function loadModelInfo() {
  const modelValue = document.getElementById('model-value');
  const engineBadge = document.getElementById('engine-badge');
  try {
    const r = await fetch(`${API}/api/config`, { signal: AbortSignal.timeout(4000) });
    if (r.ok) {
      const data = await r.json();
      if (modelValue && data.model) {
        modelValue.textContent = data.model;
      }
      if (engineBadge) {
        if (data.groq_active) {
          engineBadge.textContent = '🟢 Live Groq LLaMA-70B';
          engineBadge.style.background = 'rgba(16, 185, 129, 0.15)';
          engineBadge.style.color = '#10b981';
          engineBadge.style.border = '1px solid rgba(16, 185, 129, 0.3)';
        } else {
          engineBadge.textContent = '🟡 Grounded Clinical Engine';
          engineBadge.style.background = 'rgba(245, 158, 11, 0.15)';
          engineBadge.style.color = '#f59e0b';
          engineBadge.style.border = '1px solid rgba(245, 158, 11, 0.3)';
        }
      }
    }
  } catch (err) {
    if (modelValue) modelValue.textContent = 'Unknown';
  }
}
loadModelInfo();

/* ── DROP ZONE ─────────────────────────────────────────────────────────── */
const dropZone  = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');

if (dropZone && fileInput) {
  dropZone.addEventListener('click', () => fileInput.click());
  dropZone.addEventListener('keydown', e => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); }
  });
  dropZone.addEventListener('dragenter', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
  dropZone.addEventListener('dragover',  e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
  dropZone.addEventListener('dragleave', e => { if (!dropZone.contains(e.relatedTarget)) dropZone.classList.remove('drag-over'); });
  dropZone.addEventListener('drop', e => { e.preventDefault(); dropZone.classList.remove('drag-over'); addFiles([...e.dataTransfer.files]); });
  fileInput.addEventListener('change', () => { addFiles([...fileInput.files]); fileInput.value = ''; });
}

if (queryInput) {
  queryInput.addEventListener('input', updateRunButtonState);
}

function addFiles(files) {
  const pdfs = files.filter(f => f.name.toLowerCase().endsWith('.pdf'));
  const bad  = files.filter(f => !f.name.toLowerCase().endsWith('.pdf'));
  if (bad.length) showWarning(`${bad.map(f=>f.name).join(', ')} skipped — PDFs only.`);
  for (const f of pdfs) {
    if (queuedFiles.length >= 5) { showWarning('Max 5 PDFs.'); break; }
    if (!queuedFiles.find(q => q.name === f.name)) queuedFiles.push(f);
  }
  renderFileList();
}
function removeFile(name) { queuedFiles = queuedFiles.filter(f => f.name !== name); renderFileList(); }
function renderFileList() {
  const list = document.getElementById('file-list');
  const btnC = document.getElementById('btn-clear-files');
  updateRunButtonState();

  if (!queuedFiles.length) {
    list.classList.add('hidden');
    list.innerHTML = '';
    if (btnC) btnC.style.display = '';
    return;
  }
  list.classList.remove('hidden');
  if (btnC) btnC.style.display = '';
  list.innerHTML = queuedFiles.map(f => `
    <div class="file-item ready">
      <span class="file-item-icon">📄</span>
      <div class="file-item-info">
        <div class="file-item-name">${esc(f.name)}</div>
        <div class="file-item-meta">${fmtBytes(f.size)}</div>
      </div>
      <button class="file-item-rm" onclick="removeFile('${esc(f.name).replace(/'/g,"\\'")}')">✕</button>
    </div>`).join('');
}
document.getElementById('btn-clear-files').addEventListener('click', () => {
  queuedFiles = [];
  renderFileList();
  clearWarnings();
  updatePaperSources([]);
  if (queryInput) {
    queryInput.value = '';
    updateRunButtonState();
  }
});
function fmtBytes(b) {
  return b < 1024 ? b+'B' : b < 1048576 ? (b/1024).toFixed(1)+'KB' : (b/1048576).toFixed(1)+'MB';
}

/* ── WARNINGS ──────────────────────────────────────────────────────────── */
function showWarning(msg) {
  const el = document.getElementById('upload-warnings');
  el.classList.remove('hidden'); el.textContent = msg;
}
function clearWarnings() {
  const el = document.getElementById('upload-warnings');
  el.classList.add('hidden'); el.textContent = '';
}

/* ── TOKEN DASHBOARD ───────────────────────────────────────────────────── */
function updateTokenDashboard(meta, tokenStats) {
  const total      = meta.llm_tokens       || meta.mistral_tokens || 0;
  const calls      = meta.llm_calls        || meta.mistral_calls || 0;
  const cache      = meta.cache_hits       || 0;
  const prompt     = tokenStats?.llm_prompt         || tokenStats?.mistral_prompt || 0;
  const completion = tokenStats?.llm_completion     || tokenStats?.mistral_completion || 0;

  // Estimate tokens saved: if we had sent raw text it would be ~450 tokens/paper
  // Instead we sent struct+RAG which averages ~100 tokens total
  const papers   = Math.max(calls, 1);
  const wouldHave = papers * 450;
  const saved    = Math.max(0, wouldHave - total);

  // Animate the big number
  animCount(document.getElementById('tok-total'), total);

  setTok('tok-prompt',     prompt);
  setTok('tok-completion', completion);
  setTok('tok-calls',      calls);
  setTok('tok-cache',      cache);

  // Bar: proportion of "would-have" tokens saved vs used
  const pctUsed  = wouldHave > 0 ? Math.min(total  / wouldHave * 100, 100) : 0;
  const pctSaved = Math.max(0, 100 - pctUsed);
  document.getElementById('token-bar-saved').style.width = pctSaved.toFixed(1) + '%';
  document.getElementById('token-bar-used').style.width  = pctUsed.toFixed(1) + '%';

  document.getElementById('tok-live-badge').classList.add('active');
}
function setTok(id, val) {
  const el = document.getElementById(id);
  if (el) el.textContent = (val !== undefined && val !== null && !isNaN(val)) ? val : '—';
}

document.getElementById('btn-clear-cache').addEventListener('click', async () => {
  try {
    const r = await fetch(`${API}/api/cache/clear`, { method: 'POST' });
    const d = await r.json();
    if (!r.ok) throw new Error(d.error || 'Failed to clear cache');
    showWarning('Cache cleared.');
    setTimeout(clearWarnings, 2000);
  } catch (e) {
    showWarning('Cache clear failed: ' + e.message);
  }
});

document.getElementById('btn-export-csv').addEventListener('click', async () => {
  try {
    const r = await fetch(`${API}/api/export/csv`, { method: 'POST' });
    const d = await r.json();
    if (!r.ok) throw new Error(d.error || 'CSV export failed');
    showWarning(`CSV exported: ${d.file}`);
  } catch (e) {
    showWarning('CSV export failed: ' + e.message);
  }
});

document.getElementById('btn-export-pdf').addEventListener('click', async () => {
  try {
    const r = await fetch(`${API}/api/export/pdf`, { method: 'POST' });
    const d = await r.json();
    if (!r.ok) throw new Error(d.error || 'PDF export failed');
    showWarning(`PDF exported: ${d.file}`);
  } catch (e) {
    showWarning('PDF export failed: ' + e.message);
  }
});

/* ── RUN ───────────────────────────────────────────────────────────────── */
document.getElementById('btn-fetch').addEventListener('click', async () => {
  const query = queryInput ? queryInput.value.trim() : '';
  if (!query && !queuedFiles.length) return;

  const btn = document.getElementById('btn-fetch');
  btn.disabled = true;
  
  if (query) {
    await fetchSourcesForQuery(query);
    btn.disabled = false;
    return;
  }
  
  // Directly upload if PDF
  btn.disabled = false;
});

document.getElementById('btn-analyse').addEventListener('click', async () => {
  const btn = document.getElementById('btn-analyse');
  btn.disabled = true;
  
  if (pendingPapers.length) {
     await runAnalysis(pendingPapers);
  } else if (queuedFiles.length) {
     showLoader(); animateSteps();
     // Normal PDF upload
     try {
       const fd = new FormData();
       queuedFiles.forEach(f => fd.append('files[]', f));
       const up = await fetch(`${API}/api/upload`, { method: 'POST', body: fd });
       const ud = await up.json();
       if (!up.ok) { hideLoader(); showWarning(`Upload failed: ${ud.error||'?'}`); btn.disabled=false; return; }
       if (ud.warnings?.length) showWarning(ud.warnings.join(' | '));
       if (!ud.papers?.length)  { hideLoader(); showWarning('No text extracted.'); btn.disabled=false; return; }
       await runAnalysis(ud.papers);
     } catch(e) { hideLoader(); showWarning('Error: '+e.message); }
  }
  btn.disabled = false;
});


function showErrorState(msg) {
  hideLoader(false);
  updatePipelineStatus('Error', 'Check logs');
  document.getElementById('results-container').classList.add('hidden');
  
  const emptyState = document.getElementById('empty-state');
  emptyState.classList.remove('hidden');
  
  document.getElementById('empty-icon').innerHTML = `<svg width="48" height="48" viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="10" stroke="var(--red)" stroke-width="1.5"/><line x1="12" y1="8" x2="12" y2="12" stroke="var(--red)" stroke-width="1.5" stroke-linecap="round"/><circle cx="12" cy="16" r="1" fill="var(--red)"/></svg>`;
  document.getElementById('empty-icon').style.opacity = '1';
  document.getElementById('empty-title').textContent = 'Pipeline Interrupted';
  document.getElementById('empty-title').style.color = 'var(--red)';
  document.getElementById('empty-sub').textContent = msg;
}

function resetEmptyState() {
  document.getElementById('empty-icon').innerHTML = `<svg width="48" height="48" viewBox="0 0 48 48" fill="none"><polygon points="24,4 42,14 42,34 24,44 6,34 6,14" stroke="var(--accent)" stroke-width="1.5" fill="none" opacity="0.4"/><polygon points="24,14 33,19 33,29 24,34 15,29 15,19" stroke="var(--accent)" stroke-width="1" fill="var(--accent)" opacity="0.08"/></svg>`;
  document.getElementById('empty-icon').style.opacity = '0.5';
  document.getElementById('empty-title').textContent = 'No analysis yet';
  document.getElementById('empty-title').style.color = 'var(--text-2)';
  document.getElementById('empty-sub').textContent = 'Enter a query and run the pipeline to fetch papers, then see claims, assumptions, gaps and the Epistemic Dependency Graph.';
}

async function runAnalysis(papers) {
  resetEmptyState();
  updatePipelineStatus('Running', 'Claim extraction');
  showLoader(); animateSteps();
  switchTab('claims');
  try {
    const r = await fetch(`${API}/api/analyse`, {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ papers }),
    });
    if (!r.ok) throw new Error(await r.text());
    lastResult = await r.json();
    const papersForContext = papers.map(p => ({
      id: p.id || p.paper_id || 'paper',
      paper_id: p.paper_id || p.id || 'paper',
      title: p.title || p.id || 'paper',
      source: p.source || ((p.url || p.pdf_url) ? 'retrieved' : 'upload'),
      url: p.url || p.paper_url || p.pdf_url || (p.doi ? `https://doi.org/${p.doi}` : ''),
      pdf_url: p.pdf_url || p.url || '',
      year: p.year,
      doi: p.doi || '',
      score: p.score,
      abstract: p.abstract || p.summary || p.text || '',
      summary: p.summary || '',
    }));
    lastResult.query_context = {
      papers: papersForContext,
      pipeline_papers: papersForContext,
      retrieved_count: papers.length,
      used_in_pipeline: papers.length,
      domains: [...new Set(papersForContext.map(p => p.source).filter(Boolean))],
    };
    renderResults(lastResult);
    updatePipelineStatus('Complete', 'Results ready');
  } catch (e) {
    updatePipelineStatus('Error', 'Check logs');
    showWarning('Analysis failed: '+e.message);
    showErrorState(e.message);
  }
}

async function fetchSourcesForQuery(query) {
  resetEmptyState();
  updatePipelineStatus('Searching', 'Retrieval');
  showLoader();
  try {
    const requestedTopK = parseInt(topKInput?.value || '5', 10);
    const topKPerSource = Number.isFinite(requestedTopK) ? Math.max(1, Math.min(20, requestedTopK)) : 5;
    
    // Quick log update
    const logDiv = document.getElementById('live-logs');
    if(logDiv) logDiv.textContent = 'Searching sources...';

    const r = await fetch(`${API}/api/search`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, top_k_per_source: topKPerSource }),
    });
    const body = await r.json();
    if (!r.ok) throw new Error(body.error || 'Search failed');
    
    hideLoader();
    pendingPapers = body.papers || [];
    
    const rc = document.getElementById('results-container');
    rc.classList.remove('hidden');
    switchTab('sources');
    updatePaperSources(pendingPapers, true);
    updatePipelineStatus('Ready', 'Review sources');
    
    // Switch buttons
    document.getElementById('btn-fetch').style.display = 'none';
    document.getElementById('btn-analyse').style.display = 'flex';
    document.getElementById('btn-analyse').disabled = false;
    
  } catch (e) {
    updatePipelineStatus('Error', 'Search failed');
    hideLoader();
    showWarning('Search failed: ' + e.message);
    showErrorState(e.message);
  }
}

window.removeSource = function(id) {
  pendingPapers = pendingPapers.filter(p => p.id !== id);
  updatePaperSources(pendingPapers, true);
  if (!pendingPapers.length) {
    document.getElementById('btn-analyse').disabled = true;
  }
};


/* ── LOADER ────────────────────────────────────────────────────────────── */
let _stepTimer = null;
function showLoader() {
  document.getElementById('empty-state').classList.add('hidden');
  document.getElementById('results-container').classList.add('hidden');
  document.getElementById('loader').classList.remove('hidden');
}
function hideLoader(markCompleted = true) {
  try { if (_logStream) { _logStream.close(); _logStream = null; } } catch (e) {}
  clearInterval(_stepTimer);
  if (markCompleted) {
    document.querySelectorAll('.ps').forEach(s => { s.classList.remove('active'); s.classList.add('done'); });
  } else {
    document.querySelectorAll('.ps').forEach(s => { s.classList.remove('active', 'done'); });
  }
  setTimeout(() => document.getElementById('loader').classList.add('hidden'), 350);
}
let _logStream = null;

function animateSteps() {
  document.querySelectorAll('.ps').forEach(s => s.classList.remove('active','done'));
  if (_logStream) _logStream.close();
  clearInterval(_stepTimer);
  
  const logDiv = document.getElementById('live-logs');
  if (logDiv) logDiv.textContent = 'Connecting log stream...';

  _logStream = new EventSource(`${API}/api/logs/stream`);
  _logStream.onmessage = (e) => {
      const line = e.data || '';
      if (logDiv) logDiv.textContent = line;
      
      const lower = line.toLowerCase();
      if (lower.includes("extracting claims") || lower.includes("agent 1")) highlightStep("step-1");
      else if (lower.includes("agent 2") || lower.includes("finding evidence")) highlightStep("step-2");
      else if (lower.includes("agent 3") || lower.includes("normalizing")) highlightStep("step-3");
      else if (lower.includes("agent 6.1") || lower.includes("verifying")) highlightStep("step-6v");
      else if (lower.includes("agent 6") || lower.includes("assumption generation")) highlightStep("step-6");
      else if (lower.includes("agent 4") || lower.includes("consensus")) highlightStep("step-4");
      else if (lower.includes("agent 5") || lower.includes("gaps") || lower.includes("uncertainty")) highlightStep("step-5");
  };
  
  // Make sure we stop stream when we transition out of loader
  const oldHide = window.hideLoader;
  if (!oldHide) {
    window.hideLoader = function() {
        document.getElementById('loader').classList.add('hidden');
        if (_logStream) { _logStream.close(); _logStream = null; }
    }
  }
}

function highlightStep(id) {
    const el = document.getElementById(id);
    if (!el || el.classList.contains('active')) return;
    document.querySelectorAll('.ps.active').forEach(s => { s.classList.remove('active'); s.classList.add('done'); });
    el.classList.remove('done');
    el.classList.add('active');

    const stageMap = {
      'step-1': 'Claim extraction',
      'step-2': 'Evidence attribution',
      'step-3': 'Normalisation',
      'step-6': 'Assumption extraction',
      'step-6v': 'Assumption verification',
      'step-4': 'Agreement reasoning',
      'step-5': 'Gap detection',
    };
    updatePipelineStatus('Running', stageMap[id] || 'Processing');
}

/* ── RENDER ────────────────────────────────────────────────────────────── */
function renderResults(data) {
  hideLoader();
  const rc = document.getElementById('results-container');
  rc.style.opacity = '0';
  rc.classList.remove('hidden');
  requestAnimationFrame(() => { rc.style.transition='opacity 0.4s ease'; rc.style.opacity='1'; });

  const meta = data.meta || {};
  animCount(document.getElementById('stat-claims'),        meta.total_claims   || 0);
  animCount(document.getElementById('stat-gaps'),          meta.total_gaps     || 0);
  document.getElementById('stat-time').textContent    = (meta.elapsed_sec||0)+'s';
  document.getElementById('stat-guards').textContent  = meta.hallucination_guards||0;
  document.getElementById('stat-loss').textContent    = meta.epistemic_loss !== undefined
    ? meta.epistemic_loss.toFixed(2) : '—';
  // ACE rejected label in token dashboard
  const aceEl = document.getElementById('tok-ace');
  if (aceEl) aceEl.textContent = (data.ace_report?.rejected || 0);

  // Tab count badges
  const bSources = document.getElementById('badge-sources');
  const bClaims = document.getElementById('badge-claims');
  const bAgreements = document.getElementById('badge-agreements');
  const bGaps = document.getElementById('badge-gaps');
  const pCount = (data.query_context?.pipeline_papers || data.query_context?.papers || pendingPapers || []).length;
  if (bSources) bSources.textContent = pCount ? `(${pCount})` : '';
  if (bClaims) bClaims.textContent = (data.claims||[]).length ? `(${(data.claims||[]).length})` : '';
  if (bAgreements) bAgreements.textContent = (data.agreements||[]).length ? `(${(data.agreements||[]).length})` : '';
  if (bGaps) bGaps.textContent = (data.gaps||[]).length ? `(${(data.gaps||[]).length})` : '';

  updateTokenDashboard(meta, data.token_stats);
  updatePaperSources(data.query_context?.pipeline_papers || data.query_context?.papers || []);
  renderDiagnostics(data);
  renderHAL(data.hallucination_report || {});
  renderClaims(data.claims     || []);
  renderAgreements(data.agreements || [], data.claims || []);
  renderGaps(data.gaps         || [], data.graph     || {});
  renderEDG(data.graph         || {});
}

function updatePaperSources(papers, allowRemoval = false) {
  const list = document.getElementById('sources-list');
  const rows = Array.isArray(papers) ? papers : [];
  const bSources = document.getElementById('badge-sources');
  if (bSources) bSources.textContent = rows.length ? `(${rows.length})` : '';

  if (!list) return;

  if (!rows.length) {
    list.innerHTML = '<div class="empty-sub" style="margin-top:20px;">No sources were analyzed.</div>';
    return;
  }

  // Sort them just in case (though backend already sorts them)
  const sorted = [...rows].sort((a,b) => (b.score || 0) - (a.score || 0));

  list.innerHTML = sorted.map((p, idx) => {
    const title = p.title || p.id || `Paper ${idx + 1}`;
    const paperId = p.paper_id || p.id || '';
    const source = p.source ? p.source.toUpperCase() : 'UNKNOWN';
    const year = p.year ? ` · ${p.year}` : '';
    const score = p.score != null ? p.score.toFixed(2) : '—';
    const link = p.url || p.pdf_url || (p.doi ? `https://doi.org/${p.doi}` : '');
    const cardAttrs = link ? `data-href="${esc(link)}" role="link" tabindex="0" aria-label="Open paper ${esc(title)}"` : '';
    
    let titleHtml = `<span>${esc(title)}</span>`;
    if (link) {
      titleHtml = `<a href="${esc(link)}" target="_blank" rel="noopener noreferrer">${esc(title)}</a>`;
    }

    const snippet = p.abstract || p.summary || '';
    const removeBtn = allowRemoval 
        ? `<button class="source-exclude-btn" title="Exclude this paper" onclick="removeSource('${p.id}')">×</button>` 
        : '';
    
    return `
      <div class="source-card${link ? ' is-clickable' : ''}" data-src="${esc((p.source||'').toLowerCase())}" ${cardAttrs}>
        <div class="source-card-header">${removeBtn}</div>
        <div class="source-score-col">
          <span class="source-score-val">${score}</span>
          <span class="source-score-lbl">Score</span>
        </div>
        <div class="source-content-col">
          <div class="source-title-row">
            <div class="source-title">${titleHtml}</div>
            ${paperId ? `<span class="source-id-chip" title="${esc(paperId)}">${esc(String(paperId).slice(0, 28))}</span>` : ''}
          </div>
          <div class="source-meta">
            <span class="accent">${esc(source)}</span>${esc(year)}
          </div>
          ${snippet ? `<div class="source-snippet">${esc(snippet)}</div>` : ''}
        </div>
      </div>
    `;
  }).join('');
}

const sourcesList = document.getElementById('sources-list');
if (sourcesList) {
  sourcesList.addEventListener('click', (event) => {
    const card = event.target.closest('.source-card[data-href]');
    if (!card || event.target.closest('a, button')) return;
    const href = card.dataset.href;
    if (href) window.open(href, '_blank', 'noopener,noreferrer');
  });

  sourcesList.addEventListener('keydown', (event) => {
    if (event.key !== 'Enter' && event.key !== ' ') return;
    const card = event.target.closest('.source-card[data-href]');
    if (!card) return;
    event.preventDefault();
    const href = card.dataset.href;
    if (href) window.open(href, '_blank', 'noopener,noreferrer');
  });
}

document.querySelectorAll('.sf-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.sf-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    const target = btn.dataset.src;
    document.querySelectorAll('.source-card').forEach(card => {
      card.style.display = (target === 'all' || card.dataset.src === target) ? 'flex' : 'none';
    });
  });
});

// Polyfill switchTab specifically for the manual call generated above
function switchTab(tabName) {
  document.querySelectorAll('.tab-btn').forEach(b => {
      if (b.dataset.tab === tabName) { b.classList.add('active'); }
      else { b.classList.remove('active'); }
  });
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.add('hidden'));
  const t = document.getElementById('tab-'+tabName);
  if (t) t.classList.remove('hidden');
  if (tabName === 'graph' && lastResult) {
    renderEDG(lastResult.graph || {});
    renderEDGAnalytics(lastResult.graph || {});
    renderEDGFormal(lastResult.graph || {});
  }
}

function renderDiagnostics(data) {
  const setJson = (id, value) => {
    const el = document.getElementById(id);
    if (!el) return;
    if (!value || (Array.isArray(value) && value.length === 0) || (typeof value === 'object' && !Array.isArray(value) && Object.keys(value).length === 0)) {
      el.textContent = 'None';
      return;
    }
    el.textContent = JSON.stringify(value, null, 2);
  };

  setJson('diag-query-context', data.query_context || {});
  setJson('diag-ace', data.ace_report || {});
  setJson('diag-tokens', data.token_stats || {});
  setJson('diag-errors', data.errors || []);
}

/* HAL */
function renderHAL(hr) {
  const panel = document.getElementById('hal-report');
  const grid  = document.getElementById('hal-grid');
  if (!hr.total_interventions) { panel.classList.add('hidden'); return; }
  panel.classList.remove('hidden');
  const vecs = [
    {k:'v1_claims_dropped',       l:'[V1] Claims dropped',      d:'Subject/object not traceable to source'},
    {k:'v2_spans_removed',        l:'[V2] Spans removed',        d:'Evidence spans not found in source'},
    {k:'v3_reasons_rewritten',    l:'[V3] Reasons rewritten',    d:'Agreement reasons not grounded in claims'},
    {k:'v4_gaps_dropped',         l:'[V4] Gaps dropped',         d:'Gaps with no overlap with claims'},
    {k:'v5_assumptions_rejected', l:'[V5] Assumptions rejected', d:'Failed 4-tier grounding check'},
  ];
  grid.innerHTML = vecs.map(v => {
    const val = hr[v.k] || 0;
    return `<div class="hal-cell">
      <div class="hal-cell-lbl">${esc(v.l)}</div>
      <div class="hal-cell-val ${val>0?'hit':''}">${val}</div>
      <div class="hal-cell-desc">${esc(v.d)}</div>
    </div>`;
  }).join('');
}

/* Claims */
function verificationIcon(v) {
  if (v === 'VERIFIED') return '<span class="ver-icon ver-ok">✓</span>';
  if (v === 'WEAK')     return '<span class="ver-icon ver-weak">~</span>';
  return                       '<span class="ver-icon ver-rej">✗</span>';
}

function renderClaims(claims) {
  const el = document.getElementById('claims-list');
  if (!claims.length) { el.innerHTML='<div class="empty-msg">No claims extracted.</div>'; return; }
  el.innerHTML = claims.map(c => {
    const assumptions = c.assumptions || [];
    const rel = (c.provenance && c.provenance.study_reliability !== undefined) ? c.provenance.study_reliability : (c.study_reliability || 0.85);
    const tier = (c.provenance && c.provenance.evidence_tier) || 'Tier 1 · RCT';
    const isQuarantined = (c.provenance && c.provenance.quarantined);
    const factors = (c.provenance && c.provenance.reliability_factors) || {};
    const factorEntries = Object.entries(factors);
    const factorList = factorEntries.length
      ? factorEntries.map(([k,v]) => `<div>${esc(k.replace(/_/g,' '))}</div><div>${typeof v === 'number' ? (v>=0?'+':'')+v.toFixed(2) : esc(String(v))}</div>`).join('')
      : '';

    const assumptionBlock = assumptions.length
      ? `<div class="assumption-section">
          <div class="assumption-label">Assumptions (${assumptions.length})</div>
          <div class="assumption-rows">
            ${assumptions.map(a => `
              <div class="assumption-row">
                <div class="assumption-row-left">
                  ${verificationIcon(a.verification)}
                  <span class="assumption-type">${esc(a.type||'')}</span>
                </div>
                <div class="assumption-constraint">${esc(a.constraint||'')}</div>
                <div class="assumption-meta">
                  ${a.explicit ? '<span class="explicit-tag">explicit</span>' : '<span class="implicit-tag">implicit</span>'}
                  ${a.score ? `<span class="score-tag">${(a.score*100).toFixed(0)}%</span>` : ''}
                </div>
              </div>`).join('')}
          </div>
        </div>`
      : `<div class="assumption-section assumption-none">
          <span class="assumption-label">No assumptions extracted</span>
          <span class="ungrounded-tag">UNGROUNDED</span>
        </div>`;

    return `<div class="claim-card">
      <div class="claim-header">
        <span class="claim-paper">${esc(c.paper_source || (c.paper_id || '').slice(0,18))}</span>
        <span class="claim-text">${esc(c.text || c.subject+' '+c.predicate+' '+c.object)}</span>
      </div>
      <div class="claim-meta">
        ${c.paper_url ? `<a class="meta-pill paper-link" href="${esc(c.paper_url)}" target="_blank" rel="noopener noreferrer">open paper</a>` : ''}
        <span class="meta-pill ${isQuarantined ? 'ev-low' : (rel>=0.7?'ev-high':'ev-medium')}">
          ρ = ${rel.toFixed(2)} · ${esc(tier)} ${isQuarantined ? '(QUARANTINED)' : ''}
        </span>
        ${c.domain ? `<span class="meta-pill">${esc(c.domain)}</span>` : ''}
        ${c.method ? `<span class="meta-pill">${esc(c.method)}</span>` : ''}
      </div>
      ${factorList ? `
      <details class="rel-breakdown-card" style="margin: 6px 0;">
        <summary style="cursor:pointer;font-weight:600;font-size:10.5px;color:var(--accent);">🔍 8-Factor Reliability Breakdown (ρ = ${rel.toFixed(2)})</summary>
        <div class="rel-factor-grid" style="margin-top:6px;">
          ${factorList}
        </div>
      </details>` : ''}
      ${assumptionBlock}
      <div class="ubar-wrap">
        <div class="ubar-track"><div class="ubar-fill" style="width:${Math.round((c.uncertainty||0)*100)}%"></div></div>
        <div class="ubar-label">uncertainty: ${((c.uncertainty||0)*100).toFixed(0)}%</div>
      </div>
    </div>`;
  }).join('');
}

/* Agreements */
function basisExplanation(basis, shared) {
  const map = {
    'identical-sets':      'Identical assumption sets — same epistemic context',
    'disjoint-sets':       'Disjoint assumption sets — incompatible contexts',
    'partial-overlap':     'Partial assumption overlap — context-dependent',
    'predicate-heuristic': 'Determined by predicate opposition',
    'path-inference':      'Inferred via EDG shortest path',
    'pico-reliability-weighted': 'PICO reliability-weighted consensus',
    'no-assumptions':      'No assumptions on either claim',
    'default':             'No structural signal found',
  };
  let s = map[basis] || basis || '';
  if (shared && shared.length) {
    s += ` · shared: ${shared.slice(0,2).map(x => '<span class="shared-tag">'+esc(x)+'</span>').join(' ')}`;
  }
  return s;
}

function renderAgreements(agreements, claims) {
  const el = document.getElementById('agreements-list');
  const cm = Object.fromEntries(claims.map(c=>[c.id,c]));
  if (!agreements.length) { el.innerHTML='<div class="empty-msg">No agreements computed.</div>'; return; }
  el.innerHTML = agreements.map(a => {
    const ci = cm[a.claim_i_id], cj = cm[a.claim_j_id];
    const t1 = ci ? esc((ci.text||'').slice(0,65)) : a.claim_i_id;
    const t2 = cj ? esc((cj.text||'').slice(0,65)) : a.claim_j_id;
    const rawRel = a.relation || 'unrelated';
    const rel = rawRel === 'contradict' ? 'conditional' : rawRel;
    const basis = basisExplanation(a.agreement_basis, a.shared_assumptions);

    const p1 = (ci && ci.pico) || {};
    const int1 = p1.intervention || ci?.subject || '';
    const out1 = p1.outcome || ci?.object || '';
    const picoChips = int1 || out1 ? `
      <div class="ag-pico" style="margin: 4px 0;">
        ${int1 ? `<span class="pico-chip int">💊 ${esc(int1.slice(0,32))}</span>` : ''}
        ${out1 ? `<span class="pico-chip out">🎯 ${esc(out1.slice(0,32))}</span>` : ''}
      </div>` : '';

    return `<div class="agreement-card ${rel}">
      <div class="ag-top">
        <div class="rel-badge ${rel}">${rel.toUpperCase()}</div>
        <span class="ag-conf">${((a.confidence||0)*100).toFixed(0)}% conf</span>
      </div>
      ${picoChips}
      <div class="ag-claims">[C1] ${t1}…<br>[C2] ${t2}…</div>
      <div class="ag-basis">
        <span class="basis-icon">⊕</span>
        <span>${basis}</span>
      </div>
      ${a.reason && !a.reason.startsWith('set-op') && !a.reason.startsWith('predicate') && !a.reason.startsWith('default')
        ? `<div class="ag-reason">${esc(a.reason)}</div>` : ''}
    </div>`;
  }).join('');
}

/* Gaps */
function renderGaps(gaps, graph) {
  const el = document.getElementById('gaps-list');
  if (!gaps.length) {
    el.innerHTML='<div class="empty-msg">No research gaps detected.</div>'; return;
  }
  el.innerHTML = gaps.map(g => {
    const sigs  = g.gap_signals || {};
    const fired = sigs.signals_fired || [];
    const sigLabels = {
      'low connectivity': '⬡',
      'high uncertainty': '⚠',
      'weak evidence':    '◎',
      'low centrality':   '◈',
    };
    const signalTags = fired.length
      ? `<div class="gap-signals">
          <span class="gap-signals-label">Why it's a gap:</span>
          ${fired.map(s => `<span class="gap-sig-tag">${sigLabels[s]||'•'} ${esc(s)}</span>`).join('')}
        </div>`
      : '';

    const scoreBar = sigs.gap_score !== undefined
      ? `<div class="gap-score-bar">
          <div class="gap-score-fill" style="width:${Math.round((sigs.gap_score||0)*100)}%"></div>
        </div>
        <div class="gap-score-row">
          <span>gap score: ${((sigs.gap_score||0)*100).toFixed(0)}%</span>
          ${sigs.degree !== undefined ? `<span>degree: ${sigs.degree}</span>` : ''}
          ${sigs.betweenness !== undefined ? `<span>bc: ${sigs.betweenness.toFixed(3)}</span>` : ''}
          <span>uncertainty: ${((g.uncertainty_score||0)*100).toFixed(0)}%</span>
          ${sigs.evidence ? `<span>evidence: ${esc(sigs.evidence)}</span>` : ''}
        </div>`
      : `<div class="gap-score">uncertainty: ${((g.uncertainty_score||0)*100).toFixed(0)}%</div>`;

    return `<div class="gap-card">
      <div class="gap-header">
        <span class="gap-prio ${g.priority||'medium'}">${(g.priority||'MEDIUM').toUpperCase()}</span>
        <span class="gap-type">${esc(g.type||'empirical')}</span>
        ${sigs.gap_score !== undefined ? `<span class="gap-score-chip">${((sigs.gap_score||0)*100).toFixed(0)}</span>` : ''}
      </div>
      <div class="gap-text">${esc(g.gap)}</div>
      ${signalTags}
      ${scoreBar}
      <div style="margin-top: 12px; display: flex; justify-content: flex-end;">
        <button class="trace-btn btn secondary" onclick="traceGap('${g.id}')">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="margin-right: 4px; vertical-align: middle;">
            <path d="M12 20a8 8 0 1 0 0-16 8 8 0 0 0 0 16z"/>
            <path d="M12 14a2 2 0 1 0 0-4 2 2 0 0 0 0 4z"/>
            <path d="M12 4v2"/><path d="M12 18v2"/><path d="M4 12h2"/><path d="M18 12h2"/>
          </svg>
          Locate in Graph
        </button>
      </div>
    </div>`;
  }).join('');
}

function traceGap(gapId) {
    if (typeof edgZoomNode !== 'function') return;

    let targetId = gapId;
    if (!edgNodePositions[gapId]) {
      const nodes = edgGraphData?.nodes || [];
      const gapNode = nodes.find(n => n.id === gapId) || nodes.find(n => n.gap_region === true);
      if (gapNode) targetId = gapNode.id;
    }

    switchTab('graph');

    const tryZoom = (attempt = 0) => {
      if (edgNodePositions[targetId]) {
        edgZoomNode(targetId);
        return;
      }
      if (attempt >= 12) {
        edgZoomNode(targetId);
        return;
      }
      setTimeout(() => tryZoom(attempt + 1), 75 + attempt * 20);
    };

    tryZoom();
}
window.traceGap = traceGap;


/* ── EDG CANVAS ────────────────────────────────────────────────────────── */
let edgGraphData = null;
let edgNodePositions = {};
let edgTransform = { x: 0, y: 0, k: 1 };
let edgFocusNode = null;
let _isDragging = false;
let _lastDrag = { x: 0, y: 0 };
let _targetTransform = { x: 0, y: 0, k: 1 };
let _animFrame = null;
let _edgLayoutSig = '';
let _edgSettleFrames = 0;
let _edgSettled = false;

function edgHash(value) {
  const s = String(value || '');
  let h = 0;
  for (let i = 0; i < s.length; i++) h = ((h * 31) + s.charCodeAt(i)) >>> 0;
  return h;
}

function shortNodeLabel(id) {
  return String(id || '').slice(0, 6);
}

function normalizeEDGGraph(graph) {
  const rawNodes = Array.isArray(graph?.nodes) ? graph.nodes : [];
  const rawEdges = Array.isArray(graph?.edges) ? graph.edges : [];
  const nodes = rawNodes.map((node, idx) => {
    const rawId = node?.id ?? `node_${idx + 1}`;
    return { ...node, id: String(rawId) };
  });
  const nodeIds = new Set(nodes.map((node) => node.id));
  const edges = rawEdges
    .map((edge) => ({
      ...edge,
      source: String(edge?.source ?? ''),
      target: String(edge?.target ?? ''),
    }))
    .filter((edge) => nodeIds.has(edge.source) && nodeIds.has(edge.target) && edge.source && edge.target);

  return {
    ...graph,
    nodes,
    edges,
  };
}

let currentGraphFilter = 'all';
window.setGraphFilter = function(filter) {
  currentGraphFilter = filter;
  document.querySelectorAll('.filter-btn').forEach(btn => btn.classList.remove('active'));
  const activeBtn = document.getElementById(`btn-filter-${filter === 'high_rel' ? 'high-rel' : filter}`);
  if (activeBtn) activeBtn.classList.add('active');
  _edgSettled = false;
  if (!_animFrame) _animFrame = requestAnimationFrame(drawEDGLoop);
};

function renderEDG(graph) {
  edgGraphData = normalizeEDGGraph(graph);
  
  const canvas = document.getElementById('edg-canvas');
  if (!canvas) return;
  const W = canvas.width, H = canvas.height;
  const nodes = edgGraphData.nodes || [];
  const edges = edgGraphData.edges || [];
  const nextSig = JSON.stringify({
    n: nodes.map(n => `${n.id}:${n.type || ''}`).sort(),
    e: edges.map(e => `${e.source}>${e.target}:${e.relation || ''}`).sort(),
  });
  const shouldRelayout = _edgLayoutSig !== nextSig;

  if (shouldRelayout) {
    edgNodePositions = {};
    edgFocusNode = null;
    edgTransform = { x: 0, y: 0, k: 1 };
    _targetTransform = { x: 0, y: 0, k: 1 };
    _edgSettleFrames = 0;
    _edgSettled = false;
    _edgLayoutSig = nextSig;
  }
  
  if (nodes.length && shouldRelayout) {
    const pos = {};
    const claimNodes  = nodes.filter(n => n.type === 'claim');
    const assumpNodes = nodes.filter(n => n.type !== 'claim');
    
    // Setup generic circular layout relative to 0,0 
    // We will draw it at W/2, H/2 later using the camera transform
    claimNodes.forEach((n,i) => {
      const a = (2*Math.PI*i/Math.max(claimNodes.length,1)) - Math.PI/2;
      const rL = Math.min(W,H)*0.32;
      const pr = n.pagerank || 0;
      const r  = Math.max(16, Math.min(26, 18 + pr*100));
      const seed = edgHash(n.id);
      const jitterX = ((seed % 31) - 15) * 4.2;
      const jitterY = (((seed >> 5) % 31) - 15) * 4.2;
      pos[n.id] = { x: rL*Math.cos(a) + jitterX, y: rL*Math.sin(a) + jitterY, vx: 0, vy: 0, r: r, node: n };
    });
    assumpNodes.forEach((n,i) => {
      const a = (2*Math.PI*i/Math.max(assumpNodes.length,1));
      const rL = Math.min(W,H)*0.16;
      const seed = edgHash(n.id);
      const jitterX = ((seed % 29) - 14) * 3.2;
      const jitterY = (((seed >> 5) % 29) - 14) * 3.2;
      pos[n.id] = { x: rL*Math.cos(a) + jitterX, y: rL*Math.sin(a) + jitterY, vx: 0, vy: 0, r: 11, node: n };
    });
    edgNodePositions = pos;
  } else if (nodes.length) {
    // Keep references up to date when graph object updates but topology is unchanged.
    nodes.forEach((n) => {
      if (edgNodePositions[n.id]) edgNodePositions[n.id].node = n;
    });
  }
  
  if (!_animFrame) _animFrame = requestAnimationFrame(drawEDGLoop);
}

function drawEDGLoop() {
  // Smoothly interpolate camera
  edgTransform.x += (_targetTransform.x - edgTransform.x) * 0.22;
  edgTransform.y += (_targetTransform.y - edgTransform.y) * 0.22;
  edgTransform.k += (_targetTransform.k - edgTransform.k) * 0.22;
  
  // Physics Simulation Step
  if (edgGraphData && typeof edgNodePositions === 'object') {
    const nodes = edgGraphData.nodes || [];
    const edges = edgGraphData.edges || [];
    const pos = edgNodePositions;
    const ids = Object.keys(pos);

    // No physics simulation - static layout
    // Just keep nodes in their initial circular positions
    for (let i = 0; i < ids.length; i++) {
      pos[ids[i]].vx = 0;
      pos[ids[i]].vy = 0;
    }
    
    _edgSettled = true;
    
    // Smoothly track focus node if clicked
    if (edgFocusNode && edgNodePositions[edgFocusNode.id]) {
      const p = edgNodePositions[edgFocusNode.id];
      _targetTransform.x = -p.x;
      _targetTransform.y = -p.y;
      // We don't overwrite k here to allow user to scroll while focused
    }
  }
  
  _drawEDG();
  _animFrame = requestAnimationFrame(drawEDGLoop);
}

function _drawEDG() {
  const canvas = document.getElementById('edg-canvas');
  if (!canvas || !edgGraphData) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  const dark = document.documentElement.getAttribute('data-theme') !== 'light';

  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = dark ? '#13161e' : '#f6f7fb';
  ctx.fillRect(0,0,W,H);

  const nodes = edgGraphData.nodes || [], edges = edgGraphData.edges || [];
  if (!nodes.length) {
    ctx.fillStyle = dark ? '#444860' : '#9098b8';
    ctx.font = '13px Inter,sans-serif'; ctx.textAlign = 'center';
    ctx.fillText('Run analysis to see the Epistemic Dependency Graph', W/2, H/2);
    return;
  }

  const pos = edgNodePositions;
  const EC = { agree: '#4ade80', conditional:'#fbbf24', depends_on: '#c084fc', unrelated: dark?'#1e2240':'#d0d4e8' };

  ctx.save();
  ctx.translate(W/2 + edgTransform.x, H/2 + edgTransform.y);
  ctx.scale(edgTransform.k, edgTransform.k);
  
  // Get highlighted set
  let highlighted = new Set();
  if (edgFocusNode) {
      highlighted.add(edgFocusNode.id);
      edges.forEach(e => {
          if (e.source === edgFocusNode.id) highlighted.add(e.target);
          if (e.target === edgFocusNode.id) highlighted.add(e.source);
      });
  }

  // Edges
  edges.forEach(e => {
    const f = pos[e.source], t = pos[e.target];
    if (!f || !t) return;
    
    // Dim unrelated edge if focus is active
    let dimEdge = edgFocusNode && !(highlighted.has(e.source) && highlighted.has(e.target));
    if (dimEdge) return; // Skip drawing for extreme clarity, or draw very faint

    if (currentGraphFilter === 'agreed' && e.relation !== 'agree' && e.relation !== 'AGREE') return;

    ctx.beginPath(); ctx.moveTo(f.x, f.y); ctx.lineTo(t.x, t.y);
    const relation = e.relation === 'contradict' ? 'conditional' : e.relation;
    ctx.strokeStyle = EC[relation] || EC.unrelated;
    ctx.lineWidth   = (relation === 'depends_on' ? 1 : 1.8) / edgTransform.k;
    ctx.globalAlpha = relation === 'unrelated' ? 0.12 : 0.5;
    if (dimEdge) ctx.globalAlpha *= 0.1;
    ctx.setLineDash(relation === 'depends_on' ? [4,3] : []);
    ctx.stroke();
    ctx.globalAlpha = 1; ctx.setLineDash([]);
  });

  // Graph filter check for empty state
  const claimMap = Object.fromEntries((lastResult?.claims || []).map(c => [c.id, c]));
  const isNodeFilteredOut = (n) => {
    const claimObj = claimMap[n.id];
    const rel = (n.reliability !== undefined) ? n.reliability : ((claimObj && claimObj.provenance && claimObj.provenance.study_reliability) ?? (claimObj && claimObj.study_reliability) ?? 0.85);
    if (currentGraphFilter === 'unquarantined') {
      return (rel < 0.45 || (claimObj && claimObj.provenance && claimObj.provenance.quarantined));
    } else if (currentGraphFilter === 'high_rel') {
      return rel < 0.70;
    } else if (currentGraphFilter === 'agreed') {
      return !edges.some(e => (e.source === n.id || e.target === n.id) && (e.relation === 'agree' || e.relation === 'AGREE'));
    }
    return false;
  };

  const visibleCount = nodes.filter(n => !isNodeFilteredOut(n)).length;
  if (!visibleCount) {
    ctx.restore();
    ctx.fillStyle = dark ? '#8890b0' : '#555870';
    ctx.font = '13px JetBrains Mono,monospace'; ctx.textAlign = 'center';
    const filterLabel = currentGraphFilter === 'high_rel' ? 'High Reliability (ρ ≥ 0.70)' : currentGraphFilter === 'unquarantined' ? 'Unquarantined' : currentGraphFilter;
    ctx.fillText(`All studies in this query have ρ < 0.70 — 0 nodes match filter: "${filterLabel}"`, W/2, H/2);
    return;
  }

  // Nodes
  nodes.forEach(n => {
    const p = pos[n.id]; if (!p) return;
    if (isNodeFilteredOut(n)) return;
    const isClaim   = n.type === 'claim';
    const isGap     = n.gap_region === true;
    const r         = p.r;
    const u         = n.uncertainty || 0;
    
    let dimNode = edgFocusNode && !highlighted.has(n.id);
    ctx.globalAlpha = dimNode ? 0.15 : 1;

    if (isClaim && u > 0.45 && !dimNode) {
      const grd = ctx.createRadialGradient(p.x, p.y, r, p.x, p.y, r+16);
      grd.addColorStop(0, `rgba(248,113,113,${u*0.35})`);
      grd.addColorStop(1, 'transparent');
      ctx.beginPath(); ctx.arc(p.x, p.y, r+16, 0, Math.PI*2);
      ctx.fillStyle = grd; ctx.fill();
    }
    if (isGap && !dimNode) {
      ctx.beginPath(); ctx.arc(p.x, p.y, r+5, 0, Math.PI*2);
      ctx.strokeStyle = '#f0c060'; ctx.lineWidth = 1.5/edgTransform.k; ctx.globalAlpha = 0.6;
      ctx.setLineDash([3,3]); ctx.stroke();
      ctx.globalAlpha = 1; ctx.setLineDash([]);
    }

    ctx.beginPath(); ctx.arc(p.x, p.y, r, 0, Math.PI*2);
    const COMM_COLORS_DARK  = ['#1a2240','#1a2a1a','#2a1a2a','#2a2010','#1a2a2a','#201a2a'];
    const COMM_COLORS_LIGHT = ['#edf0fc','#edfcf0','#fcedf5','#fcf8ed','#edfdfd','#f5edfc'];
    const commIdx = (n.community !== undefined) ? (n.community % 6) : 0;
    const commFill = dark ? COMM_COLORS_DARK[commIdx] : COMM_COLORS_LIGHT[commIdx];

    ctx.fillStyle = isGap ? (dark ? '#1e1800' : '#fff8e0') : isClaim ? commFill : (dark ? '#1a1430' : '#f0ebfc');
    ctx.fill();

    ctx.strokeStyle = isGap ? '#f0c060' : isClaim ? (dark ? '#6c8ff0' : '#3b5fd4') : (dark ? '#c084fc' : '#7c3aed');
    ctx.lineWidth = 1.5/edgTransform.k; ctx.stroke();

    ctx.fillStyle = dark ? '#e2e4ef' : '#1a1d2e';
    let fSize = isClaim ? 10 : 9;
    ctx.font = `${isClaim?'600 ':''}${fSize/edgTransform.k}px JetBrains Mono,monospace`;
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText(shortNodeLabel(n.id), p.x, p.y);
    ctx.globalAlpha = 1;
  });
  ctx.restore();

  // Overlay stats
  const st = edgGraphData.stats || {};
  const an = edgGraphData.analytics || {};
  ctx.fillStyle = dark ? 'rgba(13,15,20,0.82)' : 'rgba(240,242,247,0.90)';
  ctx.beginPath(); ctx.roundRect(10, 10, 200, 88, 8); ctx.fill();
  ctx.fillStyle = dark ? '#8890b0' : '#555870';
  ctx.font = '10px JetBrains Mono,monospace'; ctx.textAlign = 'left';
  const lines = [
    `Claims:      ${st.num_claims||0}  Assumptions: ${st.num_assumptions||0}`,
    `Gaps:        ${st.gap_region_count||0}`,
    `Communities: ${an.num_communities||0}  Clusters: ${an.contra_clusters||0}`,
    `Avg U:       ${((st.avg_uncertainty||0)*100).toFixed(0)}%`,
  ];
  lines.forEach((l, i) => ctx.fillText(l, 20, 28 + i*14));
}

function edgZoomNode(nodeId) {
    const p = edgNodePositions[nodeId];
    if (!p) return;
    edgFocusNode = p.node;
  _edgSettled = false;
  _edgSettleFrames = 0;
    _targetTransform = { x: -p.x, y: -p.y, k: 1.5 };
}

window.edgZoomNode = edgZoomNode;

const edgCanvas = document.getElementById('edg-canvas');
const edgTooltip = document.getElementById('edg-tooltip');

edgCanvas.addEventListener('mousedown', e => {
  _isDragging = true;
  _lastDrag = { x: e.clientX, y: e.clientY };
  edgCanvas.style.cursor = 'grabbing';
});
edgCanvas.addEventListener('mousemove', e => {
  if (_isDragging) {
    _targetTransform.x += (e.clientX - _lastDrag.x) / _targetTransform.k;
    _targetTransform.y += (e.clientY - _lastDrag.y) / _targetTransform.k;
    _lastDrag = { x: e.clientX, y: e.clientY };
    return;
  }
  
  if (!edgNodePositions) return;
  const rect = edgCanvas.getBoundingClientRect();
  const scaleX = edgCanvas.width / rect.width;
  const scaleY = edgCanvas.height / rect.height;
  // Convert mouse coords to world space
  const mx = (e.clientX - rect.left) * scaleX;
  const my = (e.clientY - rect.top)  * scaleY;
  
  const wx = (mx - edgCanvas.width/2)/edgTransform.k - edgTransform.x;
  const wy = (my - edgCanvas.height/2)/edgTransform.k - edgTransform.y;
  
  let hoveredNode = null;
  for (const id in edgNodePositions) {
    const pos = edgNodePositions[id];
    if (Math.hypot(wx - pos.x, wy - pos.y) <= pos.r + 2) { hoveredNode = pos.node; break; }
  }

  if (hoveredNode) {
    edgCanvas.style.cursor = 'pointer';
    const ttOffsetX = 15, ttOffsetY = 15;
    edgTooltip.style.left = e.clientX + ttOffsetX + 'px';
    edgTooltip.style.top = e.clientY + ttOffsetY + 'px';
    edgTooltip.classList.remove('hidden');

    let textHTML = '';
    if (hoveredNode.type === 'claim') {
      const fullText = hoveredNode.text ? `<br><div style="margin-top:6px;font-size:11px;line-height:1.4;color:var(--text-3);max-width:280px;white-space:normal;">"${esc(hoveredNode.text)}"</div>` : '';
      textHTML = `<strong>Claim (${shortNodeLabel(hoveredNode.id)})</strong><br>
        <span style="color:var(--text-secondary)">${esc(hoveredNode.subject)}</span> <span style="color:var(--gold)">${esc(hoveredNode.predicate)}</span> <span style="color:var(--text-secondary)">${esc(hoveredNode.object)}</span>${fullText}<hr style="border:0;border-top:1px solid var(--border);margin:8px 0;">
        Domain: ${esc(hoveredNode.domain)}<br>
        Uncertainty: ${((hoveredNode.uncertainty||0)*100).toFixed(0)}%`;
    } else {
      textHTML = `<strong>Assumption (${shortNodeLabel(hoveredNode.id)})</strong><br>
        ${esc(hoveredNode.constraint)}<hr style="border:0;border-top:1px solid var(--border);margin:8px 0;">
        Type: ${esc(hoveredNode.assump_type||'implicit')}<br>
        Uncertainty: ${((hoveredNode.uncertainty||0)*100).toFixed(0)}%`;
    }
    edgTooltip.innerHTML = textHTML;
  } else {
    edgCanvas.style.cursor = 'crosshair';
    edgTooltip.classList.add('hidden');
  }
});

edgCanvas.addEventListener('click', e => {
  if (_isDragging && Math.hypot(e.clientX - _lastDrag.x, e.clientY - _lastDrag.y) > 5) return; // it was a drag
  
  const rect = edgCanvas.getBoundingClientRect();
  const scaleX = edgCanvas.width / rect.width;
  const scaleY = edgCanvas.height / rect.height;
  const mx = (e.clientX - rect.left) * scaleX;
  const my = (e.clientY - rect.top)  * scaleY;
  const wx = (mx - edgCanvas.width/2)/edgTransform.k - edgTransform.x;
  const wy = (my - edgCanvas.height/2)/edgTransform.k - edgTransform.y;
  
  let clickedNode = null;
  for (const id in edgNodePositions) {
    const pos = edgNodePositions[id];
    if (Math.hypot(wx - pos.x, wy - pos.y) <= pos.r + 2) { clickedNode = id; break; }
  }
  
  if (clickedNode) edgZoomNode(clickedNode);
  else { edgFocusNode = null; _targetTransform = {x: 0, y: 0, k: 1}; } // Reset click outside
});

edgCanvas.addEventListener('mouseup', () => _isDragging = false);
edgCanvas.addEventListener('mouseleave', () => { _isDragging = false; edgTooltip.classList.add('hidden'); });

edgCanvas.addEventListener('wheel', e => {
  e.preventDefault();
  const zoom = Math.exp(-e.deltaY * 0.001);
  _targetTransform.k *= zoom;
});

/* ── TABS ──────────────────────────────────────────────────────────────── */
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab-btn').forEach(b=>b.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach(p=>p.classList.add('hidden'));
    btn.classList.add('active');
    document.getElementById('tab-'+btn.dataset.tab).classList.remove('hidden');
    if (btn.dataset.tab==='graph' && lastResult) { renderEDG(lastResult.graph||{}); renderEDGAnalytics(lastResult.graph||{}); renderEDGFormal(lastResult.graph||{}); }
  });
});

/* ── HELPERS ───────────────────────────────────────────────────────────── */
function esc(s) {
  return String(s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function animCount(el, target) {
  if (!el) return;
  const start = parseInt(el.textContent)||0;
  if (start === target) { el.textContent = target; return; }
  const dur=600, step=16;
  let elapsed=0;
  const t = setInterval(()=>{
    elapsed += step;
    el.textContent = Math.round(start + (target-start)*Math.min(elapsed/dur,1));
    if (elapsed >= dur) clearInterval(t);
  }, step);
}

/* ── EDG ANALYTICS ──────────────────────────────────────────────────────── */
function renderEDGAnalytics(graph) {
  const an = graph.analytics || {};
  const nodes = graph.nodes || [];
  const nodeMap = Object.fromEntries(nodes.map(n=>[n.id, n]));

  // ── A: Communities ──────────────────────────────────────────────────────
  const commList = document.getElementById('comm-list');
  const commCount = document.getElementById('comm-count');
  const communities = an.communities || [];
  const themes = an.community_themes || [];

  if (commCount) commCount.textContent = communities.length || '—';

  if (commList && communities.length) {
    const COMM_COLORS = ['var(--accent)','var(--green)','var(--purple)','var(--amber)','var(--red)','var(--gold)'];
    commList.innerHTML = communities.map((cluster, idx) => {
      const theme = themes[idx] || `Cluster ${idx+1}`;
      const claimLabels = cluster
        .map(id => nodeMap[id]?.text || id)
        .filter(Boolean)
        .map(t => `<span class="comm-node">${esc(t.slice(0,35))}</span>`)
        .join('');
      return `<div class="comm-cluster">
        <div class="comm-header">
          <span class="comm-dot" style="background:${COMM_COLORS[idx%6]}"></span>
          <span class="comm-theme">${esc(theme)}</span>
          <span class="comm-size">${cluster.length} claims</span>
        </div>
        <div class="comm-nodes">${claimLabels}</div>
      </div>`;
    }).join('');
  } else if (commList) {
    commList.innerHTML = '<span class="edg-empty">No communities detected</span>';
  }

  // ── B: Influence propagation ────────────────────────────────────────────
  const infList = document.getElementById('influence-list');
  if (infList) {
    const claimNodes = nodes.filter(n => n.type==='claim' && n.influence_uncertainty !== undefined);
    if (claimNodes.length) {
      const sorted = [...claimNodes].sort((a,b) => b.influence_uncertainty - a.influence_uncertainty);
      infList.innerHTML = sorted.map(n => {
        const pct = Math.round((n.influence_uncertainty||0)*100);
        const localPct = Math.round((n.uncertainty||0)*100);
        const delta = pct - localPct;
        const deltaStr = delta > 0 ? `<span class="inf-up">+${delta}%</span>` : delta < 0 ? `<span class="inf-down">${delta}%</span>` : '<span class="inf-same">0%</span>';
        return `<div class="inf-row">
          <span class="inf-label">${esc((n.text||n.id||'').slice(0,40))}</span>
          <div class="inf-bar-wrap">
            <div class="inf-bar-local" style="width:${localPct}%"></div>
            <div class="inf-bar-prop"  style="width:${Math.max(0,pct-localPct)}%"></div>
          </div>
          <span class="inf-val">${pct}% ${deltaStr}</span>
        </div>`;
      }).join('');
    } else {
      infList.innerHTML = '<span class="edg-empty">No propagation data</span>';
    }
  }

  // ── C: Reasoning paths ──────────────────────────────────────────────────
  const pathList = document.getElementById('paths-list');
  const rpaths = (an.reasoning_paths || []).filter(p => !/CONTRADICTION/i.test(p.interpretation || ''));

  if (pathList && rpaths.length) {
    pathList.innerHTML = rpaths.map(p => {
      const interpColors = {
        'TRANSITIVE_SUPPORT':     'var(--green)',
        'ANCHOR_TO_GAP':          'var(--gold)',
        'MIXED_PATH':             'var(--text-3)',
      };
      const color = interpColors[p.interpretation] || 'var(--text-3)';
      const pathSteps = (p.path_text || p.path || []).map((step, i) => {
        const isLast = i === (p.path_text||p.path||[]).length - 1;
        return `<span class="path-step">${esc(step)}</span>${isLast ? '' : '<span class="path-arrow">→</span>'}`;
      }).join('');
      return `<div class="path-card">
        <div class="path-header">
          <span class="path-interp" style="color:${color}">${p.interpretation.replace(/_/g,' ')}</span>
          <span class="path-len">${p.length} steps</span>
        </div>
        <div class="path-chain">${pathSteps}</div>
      </div>`;
    }).join('');
  } else if (pathList) {
    pathList.innerHTML = '<span class="edg-empty">No reasoning paths found</span>';
  }
}

function renderEDGFormal(graph) {
  const st = graph.stats   || {};
  const an = graph.analytics || {};
  const set = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val; };
  set('fstat-claims',      `${st.num_claims||0} claims`);
  set('fstat-assumptions', `${st.num_assumptions||0} assumptions`);
  set('fstat-edges',       `${st.num_edges||0} edges`);
  set('fstat-communities', `${an.num_communities||0} communities`);
}
