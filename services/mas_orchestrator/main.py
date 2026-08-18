"""Sequential GLAS-Med orchestrator with Redis checkpoints and Kafka events."""
from pydantic import BaseModel
from services.common.app import service_app

app = service_app("MAS Orchestrator")


class PipelineRequest(BaseModel):
    query: str
    paper_ids: list[str] = []


@app.post("/runs")
def start_run(request: PipelineRequest):
    run_id = f"run-{abs(hash((request.query, tuple(request.paper_ids))))}"
    event = {"run_id": run_id, "stage": "agent_1_claim_extraction", "query": request.query, "paper_ids": request.paper_ids}
    try:
        import redis
        from services.common.settings import settings
        redis.from_url(settings.redis_url).setex(f"glas-med:{run_id}", 3600, "agent_1_claim_extraction")
    except Exception:
        pass
    try:
        from kafka import KafkaProducer
        from services.common.settings import settings
        KafkaProducer(bootstrap_servers=settings.kafka_bootstrap_servers, value_serializer=lambda value: str(value).encode()).send("glas-med.pipeline", event)
    except Exception:
        pass
    return event
