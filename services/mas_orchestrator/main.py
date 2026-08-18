"""Sequential GLAS-Med MAS Orchestrator with Redis checkpoints and Kafka events.

Coordinates the 5-agent sequential execution lifecycle:
  1. Agent 1 (Claim Extraction / BioBERT)
  2. Agent 2 (Evidence Mapping & Oxford Tiering)
  3. Agent 3 (8-Factor Reliability Scoring rho)
  4. Agent 4 (Agreement & PICO Consensus Ak)
  5. Agent 5 (Uncertainty Propagation & Gap Detection)
"""
from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from services.common.app import service_app
from services.common.settings import settings
from src.pipeline import run_pipeline
from pipeline.retrieval import Retriever

logger = logging.getLogger(__name__)
app = service_app("MAS Orchestrator")

# In-memory checkpoint fallback
_IN_MEMORY_CHECKPOINTS: Dict[str, Dict[str, Any]] = {}


class PipelineRequest(BaseModel):
    query: str = ""
    paper_ids: List[str] = []
    papers: List[Dict[str, Any]] = Field(default_factory=list)


def _get_redis():
    try:
        import redis
        client = redis.from_url(settings.redis_url, socket_timeout=1.5)
        client.ping()
        return client
    except Exception as e:
        logger.debug("Redis connection skipped: %s", e)
        return None


def _get_kafka_producer():
    try:
        from kafka import KafkaProducer
        return KafkaProducer(
            bootstrap_servers=settings.kafka_bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            request_timeout_ms=1500
        )
    except Exception as e:
        logger.debug("Kafka connection skipped: %s", e)
        return None


def checkpoint_stage(run_id: str, stage: str, status: str = "IN_PROGRESS", data: Dict[str, Any] | None = None) -> None:
    """Save pipeline checkpoint to Redis and emit event to Kafka."""
    import time
    checkpoint = {
        "run_id": run_id,
        "stage": stage,
        "status": status,
        "timestamp": time.time(),
        "data": data or {}
    }
    
    # 1. Update Redis checkpoint
    r = _get_redis()
    if r is not None:
        try:
            r.setex(f"glas-med:checkpoint:{run_id}", 7200, json.dumps(checkpoint))
        except Exception as e:
            logger.warning("Redis checkpoint failed for %s: %s", run_id, e)
    
    # 2. In-memory fallback
    _IN_MEMORY_CHECKPOINTS[run_id] = checkpoint

    # 3. Emit Kafka event
    producer = _get_kafka_producer()
    if producer is not None:
        try:
            producer.send("glas-med.pipeline.events", checkpoint)
            producer.flush()
        except Exception as e:
            logger.warning("Kafka event emission failed for %s: %s", run_id, e)


@app.post("/runs")
def start_run(request: PipelineRequest):
    """Initialize a pipeline execution run."""
    run_id = f"run-{uuid.uuid4().hex[:12]}"
    checkpoint_stage(run_id, stage="agent_1_claim_extraction", status="INITIALIZED", data={
        "query": request.query,
        "paper_ids": request.paper_ids,
        "papers_count": len(request.papers)
    })
    return {"run_id": run_id, "stage": "agent_1_claim_extraction", "status": "INITIALIZED"}


@app.post("/runs/{run_id}/execute")
def execute_five_agent_pipeline(run_id: str, request: PipelineRequest):
    """Execute the complete five-agent sequential pipeline with stage checkpoints."""
    logger.info("Executing 5-Agent sequence for run %s", run_id)
    
    # Stage 1: Claim Extraction
    checkpoint_stage(run_id, stage="agent_1_claim_extraction", status="RUNNING")
    
    papers = request.papers
    if not papers and request.query:
        retriever = Retriever(top_k_per_source=5)
        papers = retriever.retrieve(request.query, domains=["medical"])

    # Execute full pipeline with all multi-agent stages
    res_obj = run_pipeline(papers)
    result_dict = res_obj.to_dict() if hasattr(res_obj, "to_dict") else res_obj

    claims = result_dict.get("claims", [])
    agreements = result_dict.get("agreements", [])
    gaps = result_dict.get("gaps", [])

    # Stage 2: Evidence Mapping & Oxford Tiering
    checkpoint_stage(run_id, stage="agent_2_evidence_mapping", status="COMPLETED", data={"claims_count": len(claims)})

    # Stage 3: Reliability Scoring (rho)
    quarantined_count = sum(1 for c in claims if (c.get("provenance", {}).get("quarantined") or c.get("reliability", 1.0) < 0.45))
    checkpoint_stage(run_id, stage="agent_3_reliability_scoring", status="COMPLETED", data={"quarantined": quarantined_count})

    # Stage 4: Agreement & Consensus (Ak)
    checkpoint_stage(run_id, stage="agent_4_agreement_detection", status="COMPLETED", data={"agreements_count": len(agreements)})

    # Stage 5: Uncertainty Propagation & Gap Detection
    checkpoint_stage(run_id, stage="agent_5_uncertainty_propagation", status="COMPLETED", data={"gaps_count": len(gaps)})

    # Final Checkpoint
    checkpoint_stage(run_id, stage="pipeline_complete", status="SUCCESS", data={"meta": result_dict.get("meta", {})})

    return {
        "run_id": run_id,
        "status": "SUCCESS",
        "result": result_dict
    }


@app.get("/runs/{run_id}/status")
def get_run_status(run_id: str):
    """Fetch the latest live stage checkpoint for a run."""
    r = _get_redis()
    if r is not None:
        try:
            val = r.get(f"glas-med:checkpoint:{run_id}")
            if val:
                return json.loads(val.decode("utf-8"))
        except Exception as e:
            logger.warning("Redis status lookup error: %s", e)

    if run_id in _IN_MEMORY_CHECKPOINTS:
        return _IN_MEMORY_CHECKPOINTS[run_id]

    return {"run_id": run_id, "status": "NOT_FOUND"}
