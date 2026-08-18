from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    mongo_uri: str = os.getenv("MONGO_URI", "mongodb://mongodb:27017")
    neo4j_uri: str = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
    neo4j_user: str = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "glasmed")
    redis_url: str = os.getenv("REDIS_URL", "redis://redis:6379/0")
    kafka_bootstrap_servers: str = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
    biobert_model: str = os.getenv("BIOBERT_MODEL", "d4data/biomedical-ner-all")


settings = Settings()
