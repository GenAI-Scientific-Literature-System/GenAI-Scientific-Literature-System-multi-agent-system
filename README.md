# ⬡ MERLIN

**Grounded Assumption-Aware Multi-Agent Epistemic Reasoning over Scientific Literature**

> AAAI-style research system · Flask API · Interactive UI · Mistral AI (token-efficient)

---

## Architecture

```
Input Papers
   ↓
① Claim Extraction        (Agent 1  — Mistral AI)
   ↓
② Evidence Attribution    (Agent 2  — Mistral AI)
   ↓
③ Normalisation           (Agent 3  — Rule-based, 0 tokens)
   ↓
⑥ Assumption Extraction   (Agent 6  — Mistral AI)
   ↓
⑥.₁ Assumption Verification (Agent 6.1 — Local NLI / string match, 0 tokens)
   ↓
④ Agreement Reasoning     (Agent 4  — Heuristic + Mistral for ambiguous pairs)
   ↓
⑤ Uncertainty & Gaps      (Agent 5  — Formula + Mistral for gap text)
   ↓
Epistemic Dependency Graph (EDG)
```

### Formal Definitions
| Symbol | Definition |
|--------|-----------|
| `C = (S,P,O,M,D,Θ)` | Claim: Subject, Predicate, Object, Method, Domain, Params |
| `A = (type, scope, constraint, explicitness, evidence_span)` | Assumption |
| `R(Cᵢ,Cⱼ \| Aᵢ,Aⱼ)` | Conditional agreement relation |
| `U(C) = f(conflict, evidence, assumption_stability)` | Uncertainty score |
| `Gap = Region(G)` with `U(C) ≥ τ` | Research gap |

---

## Token Efficiency Strategy

| Agent | Approach | Mistral Tokens |
|-------|----------|---------------|
| Claim Extraction | Compact JSON prompt, truncated text | ~300/paper |
| Evidence Attribution | Batched claims, single call | ~250/paper |
| Normalisation | Rule-based predicate/domain map | **0** |
| Assumption Extraction | Section-level (not per-claim) | ~300/paper |
| Assumption Verification | Local NLI / string match | **0** |
| Agreement Reasoning | Heuristic pre-filter, LLM only for unknown | ~120 × ambiguous pairs |
| Gap Detection | Capped at 8 claims, single call | ~300 |

**Result: ~70–80% of decisions are made without any Mistral API call.**

---

## Quick Start

### Windows
```bat
setup.bat    # installs venv + dependencies
# Edit .env — add MISTRAL_API_KEY
run.bat      # starts server at http://localhost:5000
```

### Linux / macOS
```bash
bash setup.sh
# Edit .env — add MISTRAL_API_KEY
bash run.sh
```

---

## Project Structure

```
MERLIN/
├── config.py                   # Settings & prompt templates
├── requirements.txt
├── setup.bat / run.bat         # Windows launchers
├── setup.sh  / run.sh          # Unix launchers
├── .env.example
│
├── src/
│   ├── pipeline.py             # Main orchestrator
│   ├── mistral_client.py       # Token-efficient Mistral wrapper
│   ├── agents/
│   │   ├── agent1_claim.py
│   │   ├── agent2_evidence.py
│   │   ├── agent3_normalize.py
│   │   ├── agent6_assumption.py
│   │   ├── agent6_1_verify.py
│   │   ├── agent4_agreement.py
│   │   └── agent5_uncertainty.py
│   ├── graph/
│   │   └── edg.py              # Epistemic Dependency Graph
│   └── models/
│       └── schemas.py          # Claim, Assumption, Agreement, Gap
│
├── api/
│   └── server.py               # Flask REST API
│
├── frontend/
│   ├── index.html
│   ├── css/style.css
│   └── js/app.js
│
├── tests/
│   └── test_merlin.py          # 20+ pytest tests
│
└── data/
    └── merlin_bench_sample.json
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check |
| GET | `/api/sample` | Demo papers |
| POST | `/api/analyse` | Run full pipeline |
| POST | `/api/cache/clear` | Clear Mistral cache |

**POST `/api/analyse`**
```json
{
  "papers": [
    {"id": "paper_1", "text": "full paper text..."},
    {"id": "paper_2", "text": "another paper..."}
  ]
}
```

---

## Running Tests

```bash
# Activate venv first
pytest tests/ -v
```

---

## Configuration (`.env`)

```env
MISTRAL_API_KEY=your-key-from-console.mistral.ai
DEBUG=false
```

---

## Baselines Compared

| Feature | CoT | GraphRAG | DPR | **MERLIN** |
|---------|-----|---------|-----|-----------|
| Assumption modeling | ✗ | ✗ | ✗ | ✓ |
| Conditional reasoning | ✗ | ✗ | ✗ | ✓ |
| Anti-hallucination verification | ✗ | Partial | ✗ | ✓ |
| Uncertainty propagation | ✗ | ✗ | ✗ | ✓ |
| Research gap detection | ✗ | Partial | ✗ | ✓ |

---

*MERLIN bridges language models and scientific reasoning by modeling not just claims, but the assumptions under which they hold.*
