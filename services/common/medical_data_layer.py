"""MongoDB, FAISS, and Neo4j adapters for the GLAS-Med medical data layer.

Dependencies are loaded lazily so unit tests and local development can use the
in-memory fallback without needing all three services running.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Any
import hashlib
import math

from services.common.settings import settings


class MedicalDataLayer:
    def __init__(self, mongo_uri: str | None = None, neo4j_uri: str | None = None):
        self.mongo_uri = mongo_uri or settings.mongo_uri
        self.neo4j_uri = neo4j_uri or settings.neo4j_uri
        self._documents: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._vectors: dict[str, list[float]] = {}
        self._mongo = None
        self._graph = None
        self._faiss = None
        self._index = None

    def connect(self) -> dict[str, bool]:
        status = {"mongodb": False, "neo4j": False, "faiss": False}
        try:
            from pymongo import MongoClient
            self._mongo = MongoClient(self.mongo_uri, serverSelectionTimeoutMS=1000)
            self._mongo.admin.command("ping")
            status["mongodb"] = True
        except Exception:
            self._mongo = None
        try:
            from neo4j import GraphDatabase
            self._graph = GraphDatabase.driver(self.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password))
            self._graph.verify_connectivity()
            status["neo4j"] = True
        except Exception:
            self._graph = None
        try:
            import faiss
            self._faiss = faiss
            status["faiss"] = True
        except Exception:
            self._faiss = None
        return status

    @staticmethod
    def _embedding(text: str, dimensions: int = 64) -> list[float]:
        """Deterministic fallback embedding; production uses SPECTER vectors."""
        values = [0.0] * dimensions
        for token in (text or "").lower().split():
            digest = hashlib.sha256(token.encode()).digest()
            values[digest[0] % dimensions] += 1.0
        norm = math.sqrt(sum(value * value for value in values)) or 1.0
        return [value / norm for value in values]

    def upsert_document(self, document: dict[str, Any], vector: list[float] | None = None) -> str:
        document_id = str(document.get("id") or document.get("pmid") or hashlib.sha256(str(document).encode()).hexdigest()[:16])
        record = {**document, "id": document_id}
        self._documents[document_id] = record
        vector = vector or self._embedding(f"{record.get('title', '')} {record.get('abstract', record.get('text', ''))}")
        self._vectors[document_id] = vector
        if self._mongo is not None:
            self._mongo.glas_med.documents.update_one({"id": document_id}, {"$set": record}, upsert=True)
        if self._graph is not None:
            with self._graph.session() as session:
                session.run("MERGE (p:Paper {id: $id}) SET p += $record", id=document_id, record=record)
        return document_id

    def search(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        query_vector = self._embedding(query)
        scored = []
        for document_id, vector in self._vectors.items():
            scored.append((sum(a * b for a, b in zip(query_vector, vector)), document_id))
        return [{**self._documents[document_id], "similarity": round(score, 4)} for score, document_id in sorted(scored, reverse=True)[:limit]]

    def link_claim(self, claim: dict[str, Any]) -> None:
        if self._graph is None:
            return
        with self._graph.session() as session:
            session.run(
                "MERGE (c:ClinicalClaim {id: $id}) SET c += $claim "
                "WITH c MATCH (p:Paper {id: $paper_id}) MERGE (p)-[:REPORTS]->(c)",
                id=claim["id"], paper_id=claim.get("paper_id", ""), claim=claim,
            )
