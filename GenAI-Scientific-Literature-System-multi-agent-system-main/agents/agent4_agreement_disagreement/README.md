# 🔬 Agent 4 — Agreement & Disagreement Analyst

Part of the **Multi-Agent Research Intelligence System**

---

## Overview

Agent 4 is a standalone research intelligence module that:
- **Extracts structured claims** from scientific papers using pattern-based NLP
- **Compares claims semantically** across all paper pairs using TF-IDF cosine similarity
- **Classifies relationships** as: `agreement`, `contradiction`, `partial`, or `novel`
- **Builds a claim graph** (NetworkX) where nodes = papers, edges = relationships
- **Visualizes results** in an interactive Streamlit dashboard

---

## Architecture

```
agent4_agreement_disagreement/
│
├── agent_4_agreement_disagreement/
│   ├── agent.py          ← Claim extraction + relationship classifier
│   └── claim_graph.py    ← NetworkX graph builder
│
├── core/
│   └── orchestrator.py   ← Paper loading + result persistence
│
├── interface/
│   └── dashboard.py      ← Streamlit UI (5 views)
│
├── data/
│   ├── *.txt             ← Your paper files (plain text)
│   └── output/           ← Generated JSON results
│
├── run_agent4.py         ← Main entry point
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
1. Extract claims from all papers
2. Compare claims across all paper pairs
3. Classify relationships with confidence scores
4. Save results to `data/output/agent4_results.json`
5. Launch the dashboard at http://localhost:8501

---

## Dashboard Views

| View | Description |
|------|-------------|
| 📊 Overview | Summary metrics, claims per paper, pie chart |
| 🗺️ Agreement Map | Interactive graph: nodes=papers, edges=relationships |
| 🔍 Claim Explorer | Filter & drill into every claim comparison |
| 📈 Confidence Heatmap | Paper×Paper confidence matrix |
| 💡 Key Insights | Top agreements, contradictions, research gaps |

---

## Output Schema

Each claim comparison follows this strict schema:

```json
{
  "claim": "...",
  "papers": ["paper1", "paper2"],
  "relationship": "agreement | contradiction | partial | novel",
  "confidence": 0.0–1.0,
  "evidence": ["text span 1", "text span 2"],
  "uncertainty_score": 0.0–1.0,
  "research_gap": "...",
  "explanation": "human-readable reasoning",
  "semantic_similarity": 0.0–1.0
}
```

---

## Evaluation Metrics

- **Agreement Accuracy** = agreement_count / total_comparisons
- **Contradiction Rate** = contradiction_count / total_comparisons
- **Average Confidence** = mean confidence across all comparisons

---

## Requirements

- Python 3.9+
- streamlit
- plotly
- networkx

(All installed automatically by `setup.bat`)

---

## Research Motivation

Agent 4 addresses a core problem in science: **understanding how findings across papers relate**. 
Manual systematic reviews are expensive and slow. Agent 4 automates the first-pass analysis, 
surfacing agreements and contradictions that human reviewers might miss across large corpora.


## Web Interface (HTML Frontend)

A standalone browser-based frontend is included at `interface/dashboard.html`.

- **No installation required** — open directly in any modern browser
- Enter your Mistral API key in the header field
- Upload 2+ research papers (PDF, LaTeX, or plain text)
- Click **Run Analysis** to get expert-level agreement/disagreement analysis
- All text processing happens in-browser; only extracted text is sent to Mistral AI

This HTML frontend mirrors the full capability of the Streamlit dashboard (`interface/dashboard.py`) and can be used without Python.
