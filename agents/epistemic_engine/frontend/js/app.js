/* ═══════════════════════════════════════════════════════════════
   EPISTEMIC ENGINE — Frontend Application
   ═══════════════════════════════════════════════════════════════ */

const API = window.location.origin;
let pipelineResults = null;
let hypCount = 0;

/* ──────────────────── INIT ──────────────────── */
document.addEventListener("DOMContentLoaded", () => {
  checkHealth();
  setupTabs();
  setupModal();
  setupFilters();
  addHypothesisCard(); addHypothesisCard(); // Start with 2
  document.getElementById("addHypBtn").addEventListener("click", addHypothesisCard);
  document.getElementById("loadSampleBtn").addEventListener("click", loadSample);
  document.getElementById("runBtn").addEventListener("click", runAnalysis);
});

/* ──────────────────── HEALTH ──────────────────── */
async function checkHealth() {
  const dot = document.getElementById("statusDot");
  const txt = document.getElementById("statusText");
  try {
    const r = await fetch(`${API}/health`);
    const d = await r.json();
    if (d.api_key_set) {
      dot.classList.add("ok");
      txt.textContent = "API connected";
    } else {
      dot.classList.add("err");
      txt.textContent = "API key missing";
    }
  } catch {
    dot.classList.add("err");
    txt.textContent = "Backend offline";
  }
}

/* ──────────────────── TABS ──────────────────── */
function setupTabs() {
  document.querySelectorAll(".tab-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
      document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));
      btn.classList.add("active");
      document.getElementById(`panel-${btn.dataset.tab}`).classList.add("active");
    });
  });
}

function switchTab(name) {
  document.querySelectorAll(".tab-btn").forEach(b => {
    b.classList.toggle("active", b.dataset.tab === name);
  });
  document.querySelectorAll(".tab-panel").forEach(p => {
    p.classList.toggle("active", p.id === `panel-${name}`);
  });
}

/* ──────────────────── MODAL ──────────────────── */
function setupModal() {
  const backdrop = document.getElementById("modalBackdrop");
  const closeBtn = document.getElementById("modalClose");
  closeBtn.addEventListener("click", closeModal);
  backdrop.addEventListener("click", e => { if (e.target === backdrop) closeModal(); });
  document.addEventListener("keydown", e => { if (e.key === "Escape") closeModal(); });
}

function openModal(html) {
  document.getElementById("modalContent").innerHTML = html;
  document.getElementById("modalBackdrop").style.display = "flex";
}

function closeModal() {
  document.getElementById("modalBackdrop").style.display = "none";
}

/* ──────────────────── HYPOTHESIS CARDS ──────────────────── */
function addHypothesisCard(data = {}) {
  hypCount++;
  const n = hypCount;
  const container = document.getElementById("hypothesesContainer");
  const card = document.createElement("div");
  card.className = "hyp-card";
  card.dataset.hypId = n;
  card.innerHTML = `
    <div class="hyp-card-header">
      <span class="hyp-num">${n}</span>
      <span class="hyp-card-title">Hypothesis ${n}</span>
      <button class="hyp-remove" title="Remove" onclick="removeHyp(this)">✕</button>
    </div>
    <div class="hyp-grid">
      <div class="hyp-full">
        <label class="field-label">Hypothesis statement</label>
        <textarea class="field-textarea hyp-text" placeholder="State the scientific hypothesis clearly..." rows="2">${data.text || ""}</textarea>
      </div>
      <div>
        <label class="field-label">Paper ID</label>
        <input type="text" class="field-input hyp-paper" placeholder="paper_001" value="${data.paper_id || "paper_" + String(n).padStart(3, "0")}" />
      </div>
      <div>
        <label class="field-label">Domain</label>
        <input type="text" class="field-input hyp-domain" placeholder="e.g., cardiology" value="${data.domain || ""}" />
      </div>
      <div>
        <label class="field-label">Hypothesis ID</label>
        <input type="text" class="field-input hyp-id" placeholder="h${n}" value="${data.id || "h" + n}" />
      </div>
      <div class="hyp-half">
        <label class="field-label">Assumptions (comma-separated)</label>
        <input type="text" class="field-input hyp-assumptions" placeholder="e.g., sample size > 100, controlled environment" value="${(data.assumptions || []).join(", ")}" />
      </div>
      <div class="hyp-full">
        <label class="field-label">Variables (comma-separated)</label>
        <input type="text" class="field-input hyp-variables" placeholder="e.g., blood pressure, exercise duration, heart rate" value="${(data.variables || []).join(", ")}" />
      </div>
      <div class="hyp-full">
        <label class="field-label">Evidence summary</label>
        <input type="text" class="field-input hyp-evidence" placeholder="Brief description of supporting evidence..." value="${data.evidence || ""}" />
      </div>
    </div>
  `;
  container.appendChild(card);
}

function removeHyp(btn) {
  const card = btn.closest(".hyp-card");
  if (document.querySelectorAll(".hyp-card").length <= 2) {
    alert("At least 2 hypotheses are required.");
    return;
  }
  card.remove();
}

function getHypotheses() {
  return Array.from(document.querySelectorAll(".hyp-card")).map(card => ({
    id: card.querySelector(".hyp-id").value.trim() || `h${card.dataset.hypId}`,
    text: card.querySelector(".hyp-text").value.trim(),
    paper_id: card.querySelector(".hyp-paper").value.trim() || `paper_${card.dataset.hypId}`,
    domain: card.querySelector(".hyp-domain").value.trim() || null,
    assumptions: card.querySelector(".hyp-assumptions").value.split(",").map(s => s.trim()).filter(Boolean),
    variables: card.querySelector(".hyp-variables").value.split(",").map(s => s.trim()).filter(Boolean),
    evidence: card.querySelector(".hyp-evidence").value.trim() || null
  }));
}

/* ──────────────────── LOAD SAMPLE ──────────────────── */
async function loadSample() {
  try {
    const r = await fetch(`${API}/api/sample`);
    const d = await r.json();
    document.getElementById("contextInput").value = d.context || "";
    document.getElementById("hypothesesContainer").innerHTML = "";
    hypCount = 0;
    d.hypotheses.forEach(h => addHypothesisCard(h));
  } catch (e) {
    alert("Could not load sample: " + e.message);
  }
}

/* ──────────────────── RUN ANALYSIS ──────────────────── */
async function runAnalysis() {
  const hypotheses = getHypotheses();
  const valid = hypotheses.filter(h => h.text);
  if (valid.length < 2) {
    alert("Please fill in at least 2 hypothesis statements.");
    return;
  }

  const runBtn = document.getElementById("runBtn");
  const overlay = document.getElementById("runningOverlay");
  const stage = document.getElementById("runningStage");
  runBtn.disabled = true;
  overlay.style.display = "flex";

  const a4el  = document.getElementById("runAgent4");
  const a5el  = document.getElementById("runAgent5");
  const connEl = document.getElementById("runConnector");

  const stages = [
    { text: "Initializing pipeline...",                          a4: "active", conn: "",       a5: "" },
    { text: "Agent 4: Canonicalizing hypotheses...",             a4: "active", conn: "",       a5: "" },
    { text: "Agent 4: Simulating hypothesis worlds...",          a4: "active", conn: "",       a5: "" },
    { text: "Agent 4: Detecting compatibility...",               a4: "active", conn: "",       a5: "" },
    { text: "Agent 4 → Agent 5: Passing compatibility context...", a4: "done", conn: "active", a5: "" },
    { text: "Agent 5: Stress testing hypotheses...",             a4: "done",  conn: "active",  a5: "active" },
    { text: "Agent 5: Detecting epistemic boundaries...",        a4: "done",  conn: "active",  a5: "active" },
    { text: "Agent 5: Discovering research gaps...",             a4: "done",  conn: "active",  a5: "active" },
    { text: "Aggregating results...",                            a4: "done",  conn: "active",  a5: "done" }
  ];

  function applyStage(s) {
    stage.textContent = s.text;
    a4el.className   = "run-agent" + (s.a4   ? " " + s.a4   : "");
    connEl.className = "run-connector" + (s.conn ? " " + s.conn : "");
    a5el.className   = "run-agent" + (s.a5   ? " " + s.a5   : "");
  }

  let si = 0;
  applyStage(stages[0]);
  const stageInterval = setInterval(() => {
    si = Math.min(si + 1, stages.length - 1);
    applyStage(stages[si]);
  }, 1800);

  try {
    const payload = {
      hypotheses: valid,
      context: document.getElementById("contextInput").value.trim() || null
    };

    const r = await fetch(`${API}/api/pipeline`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    if (!r.ok) {
      const err = await r.json();
      throw new Error(err.detail || r.statusText);
    }

    pipelineResults = await r.json();
    renderResults(pipelineResults);
    switchTab("agreements");
  } catch (e) {
    alert("Analysis failed: " + e.message);
  } finally {
    clearInterval(stageInterval);
    runBtn.disabled = false;
    overlay.style.display = "none";
  }
}

/* ──────────────────── RENDER RESULTS ──────────────────── */
function renderResults(data) {
  renderAgreements(data.agreements, data.summary);
  renderUncertainties(data.uncertainties);
  renderMatrix(data.agreements);
}

/* ── Agreements ── */
function renderAgreements(agreements, summary) {
  const grid = document.getElementById("agreementsGrid");
  const sumDiv = document.getElementById("agreementSummary");

  // Summary strip
  if (summary) {
    const dist = summary.compatibility_distribution || {};
    sumDiv.innerHTML = `
      <div class="summary-item"><span class="summary-val">${summary.total_pairs_analyzed || 0}</span><span class="summary-key">Pairs</span></div>
      <div class="summary-item"><span class="summary-val">${dist.COEXISTENT || 0}</span><span class="summary-key">Coexistent</span></div>
      <div class="summary-item"><span class="summary-val">${dist.CONDITIONALLY_COMPATIBLE || 0}</span><span class="summary-key">Conditional</span></div>
      <div class="summary-item"><span class="summary-val">${dist.INCOMPATIBLE || 0}</span><span class="summary-key">Incompatible</span></div>
      <div class="summary-item"><span class="summary-val">${pct(summary.consensus_strength)}</span><span class="summary-key">Consensus</span></div>
      <div class="summary-item"><span class="summary-val">${pct(summary.average_confidence)}</span><span class="summary-key">Avg Confidence</span></div>
    `;
  }

  grid.innerHTML = "";
  if (!agreements || !agreements.length) {
    grid.innerHTML = '<div class="empty-state">No compatibility results.</div>';
    return;
  }

  agreements.forEach((a, i) => {
    const card = document.createElement("div");
    card.className = "compat-card";
    card.innerHTML = `
      <div class="compat-badge badge-${a.compatibility_type}">
        <span class="badge-dot"></span>${fmt(a.compatibility_type)}
      </div>
      <div class="compat-hyps">
        <div class="compat-hyp"><span class="paper-tag">[${a.source_references?.[0] || "H1"}]</span>${trunc(a.hypothesis_1, 110)}</div>
        <div class="compat-hyp"><span class="paper-tag">[${a.source_references?.[1] || "H2"}]</span>${trunc(a.hypothesis_2, 110)}</div>
      </div>
      <div class="compat-summary">${a.simulation_summary || ""}</div>
      <div class="compat-metrics">
        <div class="metric">
          <span class="metric-val">${score(a.world_model_divergence_score)}</span>
          <span class="metric-label">Divergence</span>
        </div>
        <div class="metric">
          <span class="metric-val">${score(a.confidence_score)}</span>
          <span class="metric-label">Confidence</span>
        </div>
      </div>
      <div class="score-bar"><div class="score-fill" style="width:${(a.world_model_divergence_score * 100).toFixed(0)}%"></div></div>
    `;
    card.addEventListener("click", () => openCompatModal(a));
    grid.appendChild(card);
  });
}

/* ── Uncertainties ── */
function renderUncertainties(uncertainties, filter = "all") {
  const list = document.getElementById("uncertaintyList");
  list.innerHTML = "";

  if (!uncertainties || !uncertainties.length) {
    list.innerHTML = '<div class="empty-state">No epistemic boundaries detected.</div>';
    return;
  }

  const filtered = filter === "all"
    ? uncertainties
    : uncertainties.filter(u => u.boundary_type === filter);

  if (!filtered.length) {
    list.innerHTML = '<div class="empty-state">No results for this filter.</div>';
    return;
  }

  filtered.forEach(u => {
    const card = document.createElement("div");
    card.className = "boundary-card";
    card.dataset.type = u.boundary_type;
    card.innerHTML = `
      <div class="boundary-header">
        <div>
          <span class="boundary-type-badge badge-${u.boundary_type}">${fmt(u.boundary_type)}</span>
          <div class="boundary-id">${u.boundary_id}</div>
        </div>
        ${u.unknown_unknown_indicator ? '<span class="unknown-flag">⚑ UNKNOWN-UNKNOWN</span>' : ""}
      </div>
      <div class="boundary-stress">${u.stress_test_summary || ""}</div>
      <div class="gap-box">
        <div class="gap-title">Research Gap</div>
        <div class="gap-text">${u.research_gap?.description || ""}</div>
      </div>
      <div class="risk-row">
        <div class="risk-gauge">
          <div class="risk-label"><span>Epistemic Risk</span><span>${score(u.epistemic_risk_score)}</span></div>
          <div class="risk-bar"><div class="risk-fill risk" style="width:${(u.epistemic_risk_score * 100).toFixed(0)}%"></div></div>
        </div>
        <div class="risk-gauge">
          <div class="risk-label"><span>Information Gain</span><span>${score(u.information_gain_score)}</span></div>
          <div class="risk-bar"><div class="risk-fill gain" style="width:${(u.information_gain_score * 100).toFixed(0)}%"></div></div>
        </div>
      </div>
    `;
    card.addEventListener("click", () => openBoundaryModal(u));
    list.appendChild(card);
  });
}

/* ── Matrix ── */
function renderMatrix(agreements) {
  const wrapper = document.getElementById("matrixWrapper");

  if (!agreements || !agreements.length) {
    wrapper.innerHTML = '<div class="empty-state">No data for matrix.</div>';
    return;
  }

  // Collect unique hypotheses
  const hypMap = new Map();
  agreements.forEach(a => {
    if (!hypMap.has(a.hypothesis_1)) hypMap.set(a.hypothesis_1, { text: a.hypothesis_1, ref: a.source_references?.[0] || "?" });
    if (!hypMap.has(a.hypothesis_2)) hypMap.set(a.hypothesis_2, { text: a.hypothesis_2, ref: a.source_references?.[1] || "?" });
  });
  const hyps = Array.from(hypMap.entries()); // [[text, {ref}]]

  const table = document.createElement("table");
  table.className = "matrix-table";

  // Header row
  const thead = table.createTHead();
  const hr = thead.insertRow();
  hr.insertCell().outerHTML = "<th></th>";
  hyps.forEach(([text, info]) => {
    const th = document.createElement("th");
    th.title = text;
    th.textContent = info.ref + " · " + trunc(text, 30);
    hr.appendChild(th);
  });

  // Body
  const tbody = table.createTBody();
  hyps.forEach(([rowText, rowInfo], ri) => {
    const tr = tbody.insertRow();
    const th = document.createElement("th");
    th.textContent = rowInfo.ref + " · " + trunc(rowText, 30);
    th.title = rowText;
    tr.appendChild(th);

    hyps.forEach(([colText], ci) => {
      const td = tr.insertCell();
      if (ri === ci) {
        td.innerHTML = `<div class="matrix-cell diagonal" style="background:var(--bg)"></div>`;
        return;
      }
      const result = agreements.find(a =>
        (a.hypothesis_1 === rowText && a.hypothesis_2 === colText) ||
        (a.hypothesis_2 === rowText && a.hypothesis_1 === colText)
      );

      if (result) {
        const color = cellColor(result.compatibility_type);
        const div = document.createElement("div");
        div.className = "matrix-cell";
        div.style.background = color;
        div.innerHTML = `
          <span class="cell-score">${result.world_model_divergence_score.toFixed(2)}</span>
          <span class="cell-type">${shortType(result.compatibility_type)}</span>
        `;
        div.addEventListener("click", () => openCompatModal(result));
        td.appendChild(div);
      } else {
        td.innerHTML = `<div class="matrix-cell diagonal"></div>`;
      }
    });
  });

  wrapper.innerHTML = "";
  wrapper.appendChild(table);

  // Legend
  const legend = document.createElement("div");
  legend.style.cssText = "display:flex;gap:1rem;flex-wrap:wrap;margin-top:1rem;font-family:var(--font-mono);font-size:0.7rem;color:var(--ink-mid)";
  legend.innerHTML = `
    <span style="display:flex;align-items:center;gap:.4rem"><span style="width:12px;height:12px;border-radius:2px;background:#2d6a4f;display:inline-block"></span>COEXISTENT</span>
    <span style="display:flex;align-items:center;gap:.4rem"><span style="width:12px;height:12px;border-radius:2px;background:#c9960c;display:inline-block"></span>CONDITIONAL</span>
    <span style="display:flex;align-items:center;gap:.4rem"><span style="width:12px;height:12px;border-radius:2px;background:#8b2020;display:inline-block"></span>INCOMPATIBLE</span>
    <span style="display:flex;align-items:center;gap:.4rem"><span style="width:12px;height:12px;border-radius:2px;background:#4a4a6a;display:inline-block"></span>UNKNOWN</span>
    <span style="color:var(--ink-light)">Cell value = world-model divergence score</span>
  `;
  wrapper.appendChild(legend);
}

/* ──────────────────── MODALS ──────────────────── */
function openCompatModal(a) {
  const html = `
    <div class="compat-badge badge-${a.compatibility_type}" style="margin-bottom:1rem">
      <span class="badge-dot"></span>${fmt(a.compatibility_type)}
    </div>
    <div class="modal-section">
      <div class="modal-label">Hypothesis 1 · ${a.source_references?.[0] || ""}</div>
      <div class="modal-text">${a.hypothesis_1}</div>
    </div>
    <div class="modal-section">
      <div class="modal-label">Hypothesis 2 · ${a.source_references?.[1] || ""}</div>
      <div class="modal-text">${a.hypothesis_2}</div>
    </div>
    <hr class="modal-divider"/>
    <div class="modal-section">
      <div class="modal-label">Simulation Summary</div>
      <div class="modal-text">${a.simulation_summary || "—"}</div>
    </div>
    ${a.conflict_basis?.length ? `
    <div class="modal-section">
      <div class="modal-label">Conflict Basis</div>
      <ul class="modal-list">${a.conflict_basis.map(c => `<li>${c}</li>`).join("")}</ul>
    </div>` : ""}
    ${a.counterfactual_analysis?.length ? `
    <div class="modal-section">
      <div class="modal-label">Counterfactual Analysis</div>
      <ul class="modal-list">${a.counterfactual_analysis.map(c => `<li>${c}</li>`).join("")}</ul>
    </div>` : ""}
    <div class="modal-section">
      <div class="modal-label">Scores</div>
      <div style="display:flex;gap:2rem;margin-top:.3rem">
        <div class="metric"><span class="metric-val">${score(a.world_model_divergence_score)}</span><span class="metric-label">Divergence</span></div>
        <div class="metric"><span class="metric-val">${score(a.confidence_score)}</span><span class="metric-label">Confidence</span></div>
      </div>
    </div>
    <hr class="modal-divider"/>
    <div class="trace-box">${JSON.stringify(a.trace)}</div>
  `;
  openModal(html);
}

function openBoundaryModal(u) {
  const html = `
    <span class="boundary-type-badge badge-${u.boundary_type}" style="margin-bottom:1rem">${fmt(u.boundary_type)}</span>
    ${u.unknown_unknown_indicator ? '<span class="unknown-flag" style="margin-left:.5rem">⚑ UNKNOWN-UNKNOWN</span>' : ""}
    <div class="modal-section" style="margin-top:1rem">
      <div class="modal-label">Boundary ID</div>
      <div class="modal-text" style="font-family:var(--font-mono);font-size:.82rem">${u.boundary_id}</div>
    </div>
    <div class="modal-section">
      <div class="modal-label">Stress Test Summary</div>
      <div class="modal-text">${u.stress_test_summary || "—"}</div>
    </div>
    ${u.failure_conditions?.length ? `
    <div class="modal-section">
      <div class="modal-label">Failure Conditions</div>
      <ul class="modal-list">${u.failure_conditions.map(f => `<li>${f}</li>`).join("")}</ul>
    </div>` : ""}
    <hr class="modal-divider"/>
    <div class="gap-box" style="margin-bottom:1.2rem">
      <div class="gap-title">Research Gap</div>
      <div class="gap-text" style="margin-bottom:.8rem">${u.research_gap?.description || "—"}</div>
      <div class="modal-label" style="margin-top:.6rem">Why Unresolved</div>
      <div class="gap-text">${u.research_gap?.reason_unresolved || "—"}</div>
      <div class="modal-label" style="margin-top:.6rem">Suggested Investigation</div>
      <div class="gap-text">${u.research_gap?.suggested_investigation || "—"}</div>
    </div>
    ${u.counterfactual_probes?.length ? `
    <div class="modal-section">
      <div class="modal-label">Counterfactual Probes</div>
      <ul class="modal-list">${u.counterfactual_probes.map(p => `<li>${p}</li>`).join("")}</ul>
    </div>` : ""}
    <div class="modal-section">
      <div class="modal-label">Risk & Gain Scores</div>
      <div style="display:flex;gap:2rem;margin-top:.3rem">
        <div class="metric"><span class="metric-val">${score(u.epistemic_risk_score)}</span><span class="metric-label">Epistemic Risk</span></div>
        <div class="metric"><span class="metric-val">${score(u.information_gain_score)}</span><span class="metric-label">Info Gain</span></div>
        <div class="metric"><span class="metric-val">${score(u.confidence_score)}</span><span class="metric-label">Confidence</span></div>
      </div>
    </div>
    <div class="modal-section">
      <div class="modal-label">Sources</div>
      <div class="modal-text" style="font-family:var(--font-mono);font-size:.8rem">${(u.source_references || []).join(", ")}</div>
    </div>
    <hr class="modal-divider"/>
    <div class="trace-box">${JSON.stringify(u.trace)}</div>
  `;
  openModal(html);
}

/* ──────────────────── FILTERS ──────────────────── */
function setupFilters() {
  document.querySelectorAll(".filter-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".filter-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      if (pipelineResults) {
        renderUncertainties(pipelineResults.uncertainties, btn.dataset.filter);
      }
    });
  });
}

/* ──────────────────── HELPERS ──────────────────── */
function trunc(s, n) {
  if (!s) return "";
  return s.length > n ? s.slice(0, n) + "…" : s;
}

function fmt(s) {
  return (s || "").replace(/_/g, " ");
}

function score(n) {
  return typeof n === "number" ? n.toFixed(2) : "—";
}

function pct(n) {
  return typeof n === "number" ? (n * 100).toFixed(0) + "%" : "—";
}

function cellColor(type) {
  const map = {
    COEXISTENT: "#2d6a4f",
    CONDITIONALLY_COMPATIBLE: "#c9960c",
    INCOMPATIBLE: "#8b2020",
    UNKNOWN: "#4a4a6a"
  };
  return map[type] || "#888";
}

function shortType(type) {
  const map = {
    COEXISTENT: "COEX",
    CONDITIONALLY_COMPATIBLE: "COND",
    INCOMPATIBLE: "INCOMPAT",
    UNKNOWN: "UNK"
  };
  return map[type] || type;
}
