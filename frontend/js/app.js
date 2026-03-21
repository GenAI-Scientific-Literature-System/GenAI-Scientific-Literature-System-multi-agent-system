/* ── MERLIN Frontend ───────────────────────────────────────────────────── */
const API = 'http://localhost:5000';
let lastResult = null;
let queuedFiles = [];

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

/* ── DROP ZONE ─────────────────────────────────────────────────────────── */
const dropZone  = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');

dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('keydown', e => {
  if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); }
});
dropZone.addEventListener('dragenter', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragover',  e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', e => { if (!dropZone.contains(e.relatedTarget)) dropZone.classList.remove('drag-over'); });
dropZone.addEventListener('drop', e => { e.preventDefault(); dropZone.classList.remove('drag-over'); addFiles([...e.dataTransfer.files]); });
fileInput.addEventListener('change', () => { addFiles([...fileInput.files]); fileInput.value = ''; });

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
  const btnR = document.getElementById('btn-analyse');
  if (!queuedFiles.length) {
    list.classList.add('hidden'); list.innerHTML = '';
    btnC.style.display = 'none'; btnR.disabled = true; return;
  }
  list.classList.remove('hidden'); btnC.style.display = ''; btnR.disabled = false;
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
  queuedFiles = []; renderFileList(); clearWarnings();
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
  const total      = meta.mistral_tokens   || 0;
  const calls      = meta.mistral_calls    || 0;
  const cache      = meta.cache_hits       || 0;
  const prompt     = tokenStats?.mistral_prompt     || 0;
  const completion = tokenStats?.mistral_completion || 0;

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
  if (el) el.textContent = val === 0 && id === 'tok-cache' ? '0' : (val || '—');
}

/* ── DEMO ──────────────────────────────────────────────────────────────── */
document.getElementById('btn-load-sample').addEventListener('click', async () => {
  try {
    const d = await (await fetch(`${API}/api/sample`)).json();
    queuedFiles = []; renderFileList(); clearWarnings();
    runAnalysis(d.papers);
  } catch { showWarning('Could not load demo — is the API running?'); }
});

/* ── RUN ───────────────────────────────────────────────────────────────── */
document.getElementById('btn-analyse').addEventListener('click', async () => {
  if (!queuedFiles.length) return;
  const btn = document.getElementById('btn-analyse');
  btn.disabled = true;
  showLoader();
  try {
    const fd = new FormData();
    queuedFiles.forEach(f => fd.append('files[]', f));
    const up = await fetch(`${API}/api/upload`, { method: 'POST', body: fd });
    const ud = await up.json();
    if (!up.ok) { hideLoader(); showWarning(`Upload failed: ${ud.error||'?'}`); btn.disabled=false; return; }
    if (ud.warnings?.length) showWarning(ud.warnings.join(' | '));
    if (!ud.papers?.length)  { hideLoader(); showWarning('No text extracted.'); btn.disabled=false; return; }
    await runAnalysis(ud.papers);
  } catch (e) { hideLoader(); showWarning('Error: '+e.message); }
  btn.disabled = false;
});

async function runAnalysis(papers) {
  showLoader(); animateSteps();
  try {
    const r = await fetch(`${API}/api/analyse`, {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ papers }),
    });
    if (!r.ok) throw new Error(await r.text());
    lastResult = await r.json();
    renderResults(lastResult);
  } catch (e) {
    hideLoader(); showWarning('Analysis failed: '+e.message);
  }
}

/* ── LOADER ────────────────────────────────────────────────────────────── */
let _stepTimer = null;
function showLoader() {
  document.getElementById('empty-state').classList.add('hidden');
  document.getElementById('results-container').classList.add('hidden');
  document.getElementById('loader').classList.remove('hidden');
}
function hideLoader() {
  clearInterval(_stepTimer);
  document.querySelectorAll('.ps').forEach(s => { s.classList.remove('active'); s.classList.add('done'); });
  setTimeout(() => document.getElementById('loader').classList.add('hidden'), 350);
}
function animateSteps() {
  const ids = ['step-1','step-2','step-3','step-6','step-6v','step-4','step-5'];
  document.querySelectorAll('.ps').forEach(s => s.classList.remove('active','done'));
  let i = 0;
  clearInterval(_stepTimer);
  _stepTimer = setInterval(() => {
    if (i > 0) document.getElementById(ids[i-1])?.classList.replace('active','done');
    if (i < ids.length) document.getElementById(ids[i++])?.classList.add('active');
    else clearInterval(_stepTimer);
  }, 700);
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
  animCount(document.getElementById('stat-contradictions'),meta.contradictions || 0);
  animCount(document.getElementById('stat-gaps'),          meta.total_gaps     || 0);
  document.getElementById('stat-time').textContent    = (meta.elapsed_sec||0)+'s';
  document.getElementById('stat-guards').textContent  = meta.hallucination_guards||0;
  document.getElementById('stat-loss').textContent    = meta.epistemic_loss !== undefined
    ? meta.epistemic_loss.toFixed(2) : '—';
  // ACE rejected label in token dashboard
  const aceEl = document.getElementById('tok-ace');
  if (aceEl) aceEl.textContent = (data.ace_report?.rejected || 0);

  updateTokenDashboard(meta, data.token_stats);
  renderHAL(data.hallucination_report || {});
  renderClaims(data.claims     || []);
  renderAgreements(data.agreements || [], data.claims || []);
  renderGaps(data.gaps         || [], data.graph     || {});
  renderEDG(data.graph         || {});
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
        <span class="claim-paper">${esc((c.paper_id||'').slice(0,18))}…</span>
        <span class="claim-text">${esc(c.text || c.subject+' '+c.predicate+' '+c.object)}</span>
      </div>
      <div class="claim-meta">
        ${c.domain ? `<span class="meta-pill">${esc(c.domain)}</span>` : ''}
        ${c.method ? `<span class="meta-pill">${esc(c.method)}</span>` : ''}
        ${c.evidence_strength ? `<span class="meta-pill ev-${c.evidence_strength}">evidence: ${esc(c.evidence_strength)}</span>` : ''}
      </div>
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
    const rel = a.relation || 'unrelated';
    const basis = basisExplanation(a.agreement_basis, a.shared_assumptions);
    return `<div class="agreement-card ${rel}">
      <div class="ag-top">
        <div class="rel-badge ${rel}">${rel.toUpperCase()}</div>
        <span class="ag-conf">${((a.confidence||0)*100).toFixed(0)}% conf</span>
      </div>
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
    </div>`;
  }).join('');
}

/* ── EDG CANVAS ────────────────────────────────────────────────────────── */
function renderEDG(graph) {
  const canvas = document.getElementById('edg-canvas');
  const ctx    = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  const dark = document.documentElement.getAttribute('data-theme') !== 'light';

  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = dark ? '#13161e' : '#f6f7fb';
  ctx.fillRect(0,0,W,H);

  const nodes = graph.nodes || [], edges = graph.edges || [];
  if (!nodes.length) {
    ctx.fillStyle = dark ? '#444860' : '#9098b8';
    ctx.font = '13px Inter,sans-serif'; ctx.textAlign = 'center';
    ctx.fillText('Run analysis to see the Epistemic Dependency Graph', W/2, H/2);
    return;
  }

  // Layout
  const pos = {};
  const claimNodes  = nodes.filter(n => n.type === 'claim');
  const assumpNodes = nodes.filter(n => n.type !== 'claim');
  claimNodes.forEach((n,i) => {
    const a = (2*Math.PI*i/Math.max(claimNodes.length,1)) - Math.PI/2;
    const r = Math.min(W,H)*0.30;
    pos[n.id] = { x: W/2 + r*Math.cos(a), y: H/2 + r*Math.sin(a) };
  });
  assumpNodes.forEach((n,i) => {
    const a = (2*Math.PI*i/Math.max(assumpNodes.length,1));
    const r = Math.min(W,H)*0.13;
    pos[n.id] = { x: W/2 + r*Math.cos(a), y: H/2 + r*Math.sin(a) };
  });

  // Colors
  const EC = {
    agree:      '#4ade80', contradict: '#f87171',
    conditional:'#fbbf24', depends_on: '#c084fc', unrelated: dark?'#1e2240':'#d0d4e8',
  };

  // Contradiction path highlight
  const cpPath = graph.analytics?.contradiction_path || [];
  if (cpPath.length > 1) {
    for (let i = 0; i < cpPath.length - 1; i++) {
      const f = pos[cpPath[i]], t = pos[cpPath[i+1]];
      if (!f || !t) continue;
      ctx.beginPath(); ctx.moveTo(f.x,f.y); ctx.lineTo(t.x,t.y);
      ctx.strokeStyle = '#f0c060'; ctx.lineWidth = 2.5;
      ctx.globalAlpha = 0.35; ctx.setLineDash([6,4]); ctx.stroke();
      ctx.globalAlpha = 1; ctx.setLineDash([]);
    }
  }

  // Edges
  edges.forEach(e => {
    const f = pos[e.source], t = pos[e.target];
    if (!f || !t) return;
    ctx.beginPath(); ctx.moveTo(f.x, f.y); ctx.lineTo(t.x, t.y);
    ctx.strokeStyle = EC[e.relation] || EC.unrelated;
    ctx.lineWidth   = e.relation === 'depends_on' ? 1 : 1.8;
    ctx.globalAlpha = e.relation === 'unrelated' ? 0.12 : 0.5;
    ctx.setLineDash(e.relation === 'depends_on' ? [4,3] : []);
    ctx.stroke();
    ctx.globalAlpha = 1; ctx.setLineDash([]);
  });

  // Nodes
  nodes.forEach(n => {
    const p = pos[n.id]; if (!p) return;
    const isClaim   = n.type === 'claim';
    const isGap     = n.gap_region === true;
    const pr        = n.pagerank || 0;
    const cl        = n.clustering !== undefined ? n.clustering : 0.5;
    const r         = isClaim ? Math.max(16, Math.min(26, 18 + pr*100)) : 11;
    // Low clustering = isolated = draw with reduced opacity ring
    const infU      = n.influence_uncertainty !== undefined ? n.influence_uncertainty : (n.uncertainty||0);
    const opacity   = isClaim ? Math.max(0.4, 1.0 - infU*0.4) : 0.7;
    const u         = n.uncertainty || 0;

    // High-uncertainty glow
    if (isClaim && u > 0.45) {
      const grd = ctx.createRadialGradient(p.x, p.y, r, p.x, p.y, r+16);
      grd.addColorStop(0, `rgba(248,113,113,${u*0.35})`);
      grd.addColorStop(1, 'transparent');
      ctx.beginPath(); ctx.arc(p.x, p.y, r+16, 0, Math.PI*2);
      ctx.fillStyle = grd; ctx.fill();
    }

    // Gap region pulse ring
    if (isGap) {
      ctx.beginPath(); ctx.arc(p.x, p.y, r+5, 0, Math.PI*2);
      ctx.strokeStyle = '#f0c060'; ctx.lineWidth = 1.5; ctx.globalAlpha = 0.6;
      ctx.setLineDash([3,3]); ctx.stroke();
      ctx.globalAlpha = 1; ctx.setLineDash([]);
    }

    ctx.beginPath(); ctx.arc(p.x, p.y, r, 0, Math.PI*2);
    // Community colours (up to 6 distinct communities)
    const COMM_COLORS_DARK  = ['#1a2240','#1a2a1a','#2a1a2a','#2a2010','#1a2a2a','#201a2a'];
    const COMM_COLORS_LIGHT = ['#edf0fc','#edfcf0','#fcedf5','#fcf8ed','#edfdfd','#f5edfc'];
    const commIdx = (n.community !== undefined) ? (n.community % 6) : 0;
    const commFill = dark ? COMM_COLORS_DARK[commIdx] : COMM_COLORS_LIGHT[commIdx];

    ctx.fillStyle = isGap
      ? (dark ? '#1e1800' : '#fff8e0')
      : isClaim ? commFill
      : (dark ? '#1a1430' : '#f0ebfc');
    ctx.fill();

    ctx.strokeStyle = isGap
      ? '#f0c060'
      : isClaim
        ? (dark ? '#6c8ff0' : '#3b5fd4')
        : (dark ? '#c084fc' : '#7c3aed');
    ctx.lineWidth = 1.5; ctx.stroke();

    ctx.fillStyle = dark ? '#e2e4ef' : '#1a1d2e';
    ctx.font = `${isClaim?'600 ':''}10px JetBrains Mono,monospace`;
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText((n.id||'').slice(0,6), p.x, p.y);
  });

  // Stats overlay
  const st = graph.stats || {};
  const overlayW = 200, overlayH = 88;
  ctx.fillStyle = dark ? 'rgba(13,15,20,0.82)' : 'rgba(240,242,247,0.90)';
  ctx.beginPath();
  ctx.roundRect(10, 10, overlayW, overlayH, 8);
  ctx.fill();
  ctx.fillStyle = dark ? '#8890b0' : '#555870';
  ctx.font = '10px JetBrains Mono,monospace'; ctx.textAlign = 'left';
  const an = graph.analytics || {};
  const lines = [
    `Claims:      ${st.num_claims||0}  Assumptions: ${st.num_assumptions||0}`,
    `Contra:      ${st.contradiction_count||0}  Gaps: ${st.gap_region_count||0}`,
    `Communities: ${an.num_communities||0}  Clusters: ${an.contra_clusters||0}`,
    `Avg U:       ${((st.avg_uncertainty||0)*100).toFixed(0)}%`,
  ];
  lines.forEach((l, i) => ctx.fillText(l, 20, 28 + i*14));
}

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
  const rpaths = an.reasoning_paths || [];

  if (pathList && rpaths.length) {
    pathList.innerHTML = rpaths.map(p => {
      const interpColors = {
        'DIRECT_CONTRADICTION':   'var(--red)',
        'INDIRECT_CONTRADICTION': 'var(--amber)',
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
