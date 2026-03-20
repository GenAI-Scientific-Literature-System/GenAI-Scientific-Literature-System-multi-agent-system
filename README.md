# Multi-Agent Generative AI System for Scientific Literature Analysis

> A modular, evidence-grounded pipeline that analyses scientific literature in response to natural language queries — retrieving, clustering, and synthesising findings across papers using a coordinated agent ensemble.

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
Similarity Graph Construction
    │
    ▼
Graph-Based Clustering
    │
    ▼
Multi-Agent Analysis (Concurrent Ensemble)
    │
    ├── Agent 1: Claim Extraction
    ├── Agent 2: Evidence Collection
    ├── Agent 3: Study Reliability
    ├── Agent 4: Agreement Detection
    └── Agent 5: Uncertainty Priority
    │
    ▼
Ranked Results
```

The architecture follows a **modular pipeline** — each stage is independently developed, tested, and integrated.

---

## Agents

| Agent | Role |
|-------|------|
| **Agent 1** | Extracts principal scientific claims from each paper |
| **Agent 2** | Collects supporting or contradicting evidence spans |
| **Agent 3** | Evaluates study reliability using methodological signals |
| **Agent 4** | Detects cross-paper agreement and disagreement |
| **Agent 5** | Prioritises unresolved questions by importance and contention |

All agents run concurrently as an ensemble over the clustered paper set.

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
