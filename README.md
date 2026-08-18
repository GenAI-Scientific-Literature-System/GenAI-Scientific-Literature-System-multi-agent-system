# GLAS-Med: Generative Literature Analysis System for Medicine

> A graph-augmented, evidence-grounded pipeline for clinical literature synthesis. It retrieves biomedical evidence, extracts verifiable clinical claims, scores study reliability, detects PICO-aligned agreement, and ranks unresolved controversies.

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Agents](#agents)
- [Dataset Preparation](#dataset-preparation)
- [Pipeline Status](#pipeline-status)
- [How This Differs from General-Purpose AI](#how-this-differs-from-general-purpose-ai)
- [Roadmap](#roadmap)
- [Team](#team)

---

## Overview

This system is designed to go beyond conversational AI by grounding every response in **actual retrieved scientific documents**. Given a natural language query, the system:

1. Encodes the query and retrieved paper abstracts into a shared semantic space.
2. Constructs a similarity graph over retrieved papers and partitions them into thematic clusters.
3. Deploys a multi-agent ensemble to extract claims, gather evidence, evaluate reliability, detect consensus/disagreement, and surface unresolved questions.

---

## System Architecture

```
User Query
    │
    ▼
Query Preprocessing
    │
    ▼
Semantic Embedding
    │
    ▼
Paper Retrieval
    │
    ▼
Agent 1: Clinical Claim Extraction + PICO Projection
    │
    ▼
Agent 2: Evidence Collection + Oxford Tier Assignment
    │
    ▼
Agent 3: Eight-Factor Study Reliability (rho)
    │
    ▼
Medical Knowledge Graph Construction
    │
    ▼
Agent 4: PICO-Clustered Reliability-Weighted Agreement (A_k)
    │
    ▼
Agent 5: Uncertainty-Impact Research Gap Priority (U_k)
    │
    ▼
Ranked Results
```

The architecture follows the paper's strictly sequenced clinical pipeline. Existing validation and provenance checks remain in place as internal safeguards; they do not replace the five clinical agents.

---

## Agents

| Agent | Role |
|-------|------|
| **Agent 1** | Extracts clinical subject-predicate-object claims and attaches PICO/provenance fields |
| **Agent 2** | Collects evidence spans and stratifies study designs into Oxford evidence tiers |
| **Agent 3** | Scores reliability from design, sample size, blinding, follow-up, statistics, sponsorship, preregistration, and journal impact |
| **Agent 4** | Computes (A_k = \sum_{c_i \in F_k^*}\rho_i\sigma_i / \sum_{c_j \in C_k}\rho_j\sigma_j) for each PICO cluster |
| **Agent 5** | Ranks unresolved clusters with (U_k = (1-A_k)\bar{\rho}_k\log(1+\bar{c}_k)) |

Low-reliability studies (rho < 0.45) stay visible in provenance but are quarantined from the Agent 4 consensus calculation.

## Local microservice stack

The paper-aligned runtime is separate from the legacy Flask dashboard under `services/`. It provides nine service boundaries: session manager, inference server, tool collector, medical data layer, Agent 1 worker, graph generator, graph collector, MAS orchestrator, and frontend. The data layer persists documents in MongoDB, represents paper-to-claim relationships in Neo4j, and exposes a FAISS-ready vector interface; it has an in-memory fallback for local unit tests.

```bash
docker compose -f docker-compose.glas-med.yml up --build
```

The data layer is exposed at `http://localhost:8001`, the orchestrator at `http://localhost:8002`, and the frontend at `http://localhost:8080`. Kubernetes definitions are in `k8s/glas-med.yaml`; build and publish the `glas-med:latest` image before applying them.

---

## Dataset Preparation

Papers are retrieved **dynamically at runtime** by querying scholarly databases. Each record includes:

- Title, abstract, keywords
- Citation information

Records are then:

1. Deduplicated and stored in a structured format
2. Preprocessed via text normalisation, tokenisation, and stopword removal
3. Encoded into semantic embeddings for downstream use

---

## How This Differs from General-Purpose AI

General-purpose conversational AI generates responses from parametric knowledge encoded during training — without reading, retrieving, or comparing actual scientific documents at inference time. This system differs in three fundamental ways:

**Evidence-grounded** — Every claim is traced to a specific retrieved paper. The system does not hallucinate citations or conflate findings across studies.

**Structured multi-perspective synthesis** — Rather than returning a single narrative answer, the system explicitly maps agreement, disagreement, and uncertainty across the literature.

**Domain-specific scientific reasoning** — The agent ensemble is designed for the epistemics of scientific discourse: evaluating methodology, replication, and evidential weight — capabilities absent in general-purpose assistants.

---

## Team

| Student ID |
|------------|
| PES1UG23CS024 |
| PES1UG23CS337 |
| PES1UG23CS500 |

---

*Generative AI Project — Progress Update*
