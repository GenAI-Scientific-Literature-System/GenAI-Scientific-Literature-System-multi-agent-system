# Epistemic Engine
**AAAI-Level Scientific Reasoning System — Agents 4 & 5**

> Hypothesis Simulation + Epistemic Boundary Analysis

---

## Architecture

```
Input (Hypotheses)
       │
       ▼
┌─────────────────────────────────────────────────────┐
│  AGENT 4: Hypothesis Compatibility Engine           │
│  • Canonicalizes each hypothesis                    │
│  • Simulates pairwise world-model compatibility     │
│  • Classifies: COEXISTENT / CONDITIONALLY_COMPAT /  │
│    INCOMPATIBLE / UNKNOWN                           │
│  • Computes divergence scores + counterfactuals     │
└─────────────────────┬───────────────────────────────┘
                      │ compatibility results
                      ▼
┌─────────────────────────────────────────────────────┐
│  AGENT 5: Epistemic Boundary & Research Gap Engine  │
│  • Stress-tests hypotheses (distribution shifts,   │
│    extreme conditions, missing variables)           │
│  • Detects generalization failures                  │
│  • Discovers "unknown-unknown" dimensions           │
│  • Ranks by epistemic risk + information gain       │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
              Aggregated Output → UI
```

---

## Quick Start

### Windows
```bat
setup.bat
run.bat
```

### Linux / Mac
```bash
chmod +x setup.sh run.sh
./setup.sh
./run.sh
```

Then open: **http://localhost:8000**

---

## Requirements

- Python 3.9+
- Mistral API key (set as `MISTRAL_API_KEY` environment variable)
- Internet connection (for Mistral API calls)

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET  | `/` | Frontend UI |
| GET  | `/health` | System health check |
| GET  | `/api/sample` | Load sample hypotheses |
| POST | `/api/pipeline` | Full Agent 4 → 5 pipeline |
| POST | `/api/agent4` | Agent 4 only |
| POST | `/api/agent5` | Agent 5 only |
| GET  | `/docs` | Swagger UI |

---

## Input Format

```json
{
  "hypotheses": [
    {
      "id": "h1",
      "text": "Your hypothesis statement",
      "paper_id": "paper_001",
      "domain": "biomedical",
      "assumptions": ["assumption 1", "assumption 2"],
      "variables": ["variable A", "variable B"],
      "evidence": "Brief description of supporting evidence"
    }
  ],
  "context": "Optional research context string"
}
```

**Constraints:** 2–8 hypotheses per request.

---

## Agent 4 Output Schema

```json
{
  "hypothesis_1": "string",
  "hypothesis_2": "string",
  "compatibility_type": "COEXISTENT | CONDITIONALLY_COMPATIBLE | INCOMPATIBLE | UNKNOWN",
  "simulation_summary": "string",
  "conflict_basis": ["string"],
  "world_model_divergence_score": 0.0,
  "counterfactual_analysis": ["string"],
  "confidence_score": 0.0,
  "source_references": ["paper_id"],
  "trace": { "component": "Agent_4", "method": "hypothesis_simulation" }
}
```

## Agent 5 Output Schema

```json
{
  "boundary_id": "string",
  "related_hypotheses": ["string"],
  "boundary_type": "GENERALIZATION_FAILURE | ASSUMPTION_BREAKDOWN | UNEXPLORED_DIMENSION",
  "failure_conditions": ["string"],
  "stress_test_summary": "string",
  "unknown_unknown_indicator": true,
  "research_gap": {
    "description": "string",
    "reason_unresolved": "string",
    "suggested_investigation": "string"
  },
  "epistemic_risk_score": 0.0,
  "information_gain_score": 0.0,
  "counterfactual_probes": ["string"],
  "confidence_score": 0.0,
  "source_references": ["paper_id"],
  "trace": { "component": "Agent_5", "method": "boundary_analysis" }
}
```

---

## UI Tabs

| Tab | Description |
|-----|-------------|
| **Input** | Define hypotheses and run analysis |
| **Agreement Explorer** | Pairwise compatibility cards with simulation reasoning |
| **Uncertainty Dashboard** | Epistemic boundaries ranked by risk and information gain |
| **Compatibility Matrix** | Divergence score heatmap across all hypothesis pairs |

---

## Confidence Scoring

```
confidence = f(
  simulation_consistency,      # LLM consistency across runs
  cross_source_agreement,      # Multiple papers supporting same claim
  conflict_intensity,          # Penalty for high divergence
  stress_test_robustness       # Stability under distribution shifts
)
```

Normalized to [0, 1]. High-conflict pairs penalized; consistent multi-source support boosted.

---

## Limitations

- Max 8 hypotheses (combinatorial: O(n²) LLM calls)
- Requires Mistral API access
- LLM non-determinism means results may vary slightly between runs
- Agent 5 generates 1–2 boundary analyses per run; increase by modifying `agent5.py`

---

## File Structure

```
epistemic-engine/
├── backend/
│   ├── main.py          # FastAPI application
│   ├── agent4.py        # Hypothesis Compatibility Engine
│   ├── agent5.py        # Epistemic Boundary Engine
│   ├── pipeline.py      # Orchestration
│   ├── models.py        # Pydantic schemas
│   └── requirements.txt
├── frontend/
│   ├── index.html
│   ├── css/style.css
│   └── js/app.js
├── setup.bat / setup.sh
├── run.bat   / run.sh
└── README.md
```

---

*Epistemic Engine — Built for AAAI-level scientific reasoning research.*
