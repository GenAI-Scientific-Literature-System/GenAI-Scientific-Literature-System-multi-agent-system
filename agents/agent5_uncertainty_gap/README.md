# 🧭 Agent 5 — Uncertainty & Research Gap Analyst

Part of the **Multi-Agent Research Intelligence System**

---

## Overview

Agent 5 is a standalone research intelligence module that:
- **Scores sentence-level uncertainty** using lexicon-based NLP (hedging, limitation, assumption patterns)
- **Detects research gap signals** (future work, open questions, unstudied problems, missing benchmarks)
- **Ranks gaps** by a composite score: Impact (45%) × Feasibility (30%) × Novelty (25%)
- **Clusters gaps** by research theme (Data, Methodology, Ethics, Scalability, etc.)
- **Detects cross-paper conflicts** in uncertainty levels
- **Visualizes everything** in an interactive 6-view Streamlit dashboard

---

## Architecture

```
agent5_uncertainty_gap/
│
├── agent_5_uncertainty_gap/
│   ├── agent.py          ← Uncertainty detection + gap signal extraction
│   └── gap_ranker.py     ← Multi-dimensional ranking + theme clustering
│
├── core/
│   └── orchestrator.py   ← Paper loading + result persistence
│
├── interface/
│   └── dashboard.py      ← Streamlit UI (6 views)
│
├── data/
│   ├── *.txt             ← Your paper files (plain text)
│   └── output/           ← Generated JSON results
│
├── run_agent5.py         ← Main entry point
├── setup.bat             ← One-click environment setup
└── run.bat               ← One-click run + dashboard launch
```

---

## Quick Start

### 1. Setup (first time only)
```
setup.bat
```

### 2. Add your papers
Place plain-text `.txt` files in the `data/` folder. Each file = one paper.

### 3. Run
```
run.bat
```

This will:
1. Analyze uncertainty in all papers
2. Extract and score research gap signals
3. Rank gaps by composite score
4. Cluster by research theme
5. Detect cross-paper conflicts
6. Save results to `data/output/agent5_results.json`
7. Launch dashboard at http://localhost:8502

---

## Dashboard Views

| View | Description |
|------|-------------|
| 🏠 Overview | Summary metrics, per-paper uncertainty bars, gap type chart |
| ❓ Research Gap Dashboard | Full ranked gap list with scores, evidence, recommendations |
| 📊 Uncertainty Heatmap | Uncertainty type × paper frequency heatmap |
| ⚡ Conflict Detector | Cross-paper uncertainty level conflicts |
| 🧩 Gap Clusters | Thematic clustering with visual breakdown |
| 📋 Full Gap Table | Sortable, downloadable table of all gaps |

---

## Gap Scoring Formula

```
Composite = Impact × 0.45 + Feasibility × 0.30 + Novelty × 0.25
```

| Dimension | Description |
|-----------|-------------|
| **Impact** | Scientific importance; boosted for healthcare, safety, benchmarks |
| **Feasibility** | Ease of addressing; adjusted for data availability, resources |
| **Novelty** | Uniqueness; penalized if gap type is over-represented |

### Priority Tiers

| Tier | Composite Score |
|------|-----------------|
| 🔴 Critical | > 0.80 |
| 🟠 High | 0.70 – 0.80 |
| 🟡 Medium | 0.55 – 0.70 |
| 🟢 Low | < 0.55 |

---

## Output Schema

Each gap follows this schema:

```json
{
  "gap_id": "G001",
  "paper_source": "paper_name",
  "gap_type": "data_gap | future_direction | open_question ...",
  "category": "Missing Data / Benchmark",
  "description": "sentence from paper",
  "trigger_phrase": "exact trigger text",
  "impact_score": 0.0–1.0,
  "feasibility_score": 0.0–1.0,
  "novelty_score": 0.0–1.0,
  "composite_score": 0.0–1.0,
  "recommendation": "actionable research suggestion",
  "priority": "HIGH | MEDIUM | LOW",
  "tier": "🔴 Critical | 🟠 High | 🟡 Medium | 🟢 Low",
  "rank": 1
}
```

---

## Uncertainty Detection Patterns

Agent 5 detects 30+ linguistic patterns across categories:

- **Epistemic hedges**: may, might, could, possibly, perhaps, suggest, indicate
- **Scope limitations**: limited, preliminary, pilot, exploratory, initial
- **Gap signals**: future work, open questions, further investigation needed
- **Methodology weaknesses**: small sample, no baseline, synthetic data, no ablation
- **Knowledge boundaries**: to the best of our knowledge, as far as we know
- **Explicit uncertainty**: unclear, unknown, uncertain, ambiguous, inconclusive

---

## Requirements

- Python 3.9+
- streamlit
- plotly
- pandas

(All installed automatically by `setup.bat`)

---

## Research Motivation

Agent 5 addresses the reproducibility crisis and research planning bottleneck in science.
By systematically identifying uncertainty signals and research gaps, it helps researchers:
1. **Prioritize** which gaps to address next
2. **Avoid** over-confident claims in their own work
3. **Design** experiments that fill the highest-impact gaps
4. **Connect** their work to the broader gap landscape

This aligns with the emphasis on rigorous, reproducible, and transparent AI research.


## Web Interface (HTML Frontend)

A standalone browser-based frontend is included at `interface/dashboard.html`.

- **No installation required** — open directly in any modern browser
- Enter your Mistral API key in the header field
- Upload 1+ research papers (PDF, LaTeX, or plain text)
- Click **Run Uncertainty Audit** to get expert-level uncertainty and gap analysis
- All text processing happens in-browser; only extracted text is sent to Mistral AI

This HTML frontend mirrors the full capability of the Streamlit dashboard (`interface/dashboard.py`) and can be used without Python.
