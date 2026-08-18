"""MongoDB, FAISS, and Neo4j adapters for the GLAS-Med medical data layer.

Provides production FAISS vector indexing with cosine/inner-product search,
MongoDB document hydration, and Neo4j knowledge graph persistence.
Includes automatic daemon connection initialization and index synchronization.
"""
from __future__ import annotations

import hashlib
import logging
import math
from collections import OrderedDict
from typing import Any, List, Dict, Optional

from services.common.settings import settings

logger = logging.getLogger(__name__)


class MedicalDataLayer:
    def __init__(self, mongo_uri: str | None = None, neo4j_uri: str | None = None, dimensions: int = 64, auto_connect: bool = True):
        self.mongo_uri = mongo_uri or settings.mongo_uri
        self.neo4j_uri = neo4j_uri or settings.neo4j_uri
        self.dimensions = dimensions
        
        self._documents: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._vectors: dict[str, list[float]] = {}
        self._id_to_int: dict[str, int] = {}
        self._int_to_id: dict[int, str] = {}
        self._next_int_id = 0
        
        self._mongo = None
        self._graph = None
        self._faiss_module = None
        self._faiss_index = None
        
        self._init_faiss()
        if auto_connect:
            self.connect()

    def _init_faiss(self) -> bool:
        try:
            import faiss
            import numpy as np
            self._faiss_module = faiss
            self._faiss_index = faiss.IndexFlatIP(self.dimensions)
            logger.info("Initialized FAISS IndexFlatIP with dimension %d", self.dimensions)
            return True
        except ImportError:
            logger.warning("FAISS not installed, using normalized in-memory vector index fallback")
            self._faiss_module = None
            self._faiss_index = None
            return False

    def connect(self) -> dict[str, bool]:
        """Establish live connections to MongoDB and Neo4j."""
        status = {"mongodb": False, "neo4j": False, "faiss": self._faiss_index is not None}
        try:
            from pymongo import MongoClient
            self._mongo = MongoClient(self.mongo_uri, serverSelectionTimeoutMS=250)
            self._mongo.admin.command("ping")
            status["mongodb"] = True
            logger.info("Connected to MongoDB at %s", self.mongo_uri)
        except Exception as e:
            logger.debug("MongoDB connection skipped: %s", e)
            self._mongo = None

        try:
            from neo4j import GraphDatabase
            self._graph = GraphDatabase.driver(
                self.neo4j_uri,
                auth=(settings.neo4j_user, settings.neo4j_password),
                connection_timeout=0.25
            )
            self._graph.verify_connectivity()
            status["neo4j"] = True
            logger.info("Connected to Neo4j at %s", self.neo4j_uri)
        except Exception as e:
            logger.debug("Neo4j connection skipped: %s", e)
            self._graph = None

        return status

    def _rebuild_faiss_index(self) -> None:
        """Rebuild FAISS index upon document updates to prevent stale embeddings."""
        if self._faiss_module is None:
            return
        import numpy as np
        self._faiss_index = self._faiss_module.IndexFlatIP(self.dimensions)
        self._id_to_int.clear()
        self._int_to_id.clear()
        self._next_int_id = 0

        if self._vectors:
            vec_list = []
            for doc_id, vec in self._vectors.items():
                int_id = self._next_int_id
                self._id_to_int[doc_id] = int_id
                self._int_to_id[int_id] = doc_id
                self._next_int_id += 1
                vec_list.append(vec)
            vec_arr = np.array(vec_list, dtype=np.float32)
            self._faiss_index.add(vec_arr)

    def _embedding(self, text: str) -> list[float]:
        """Deterministic normalized embedding representation."""
        values = [0.0] * self.dimensions
        for token in (text or "").lower().split():
            digest = hashlib.sha256(token.encode()).digest()
            values[digest[0] % self.dimensions] += 1.0
        norm = math.sqrt(sum(v * v for v in values)) or 1.0
        return [v / norm for v in values]

    def upsert_document(self, document: dict[str, Any], vector: list[float] | None = None) -> str:
        doc_id = str(document.get("id") or document.get("pmid") or hashlib.sha256(str(document).encode()).hexdigest()[:16])
        record = {**document, "id": doc_id}
        
        vec = vector or self._embedding(f"{record.get('title', '')} {record.get('abstract', record.get('text', ''))}")
        is_update = (doc_id in self._vectors)
        self._documents[doc_id] = record
        self._vectors[doc_id] = vec

        # Add or update in FAISS index
        if self._faiss_index is not None:
            import numpy as np
            if not is_update:
                int_id = self._next_int_id
                self._id_to_int[doc_id] = int_id
                self._int_to_id[int_id] = doc_id
                self._next_int_id += 1
                self._faiss_index.add(np.array([vec], dtype=np.float32))
            else:
                self._rebuild_faiss_index()

        # Persist to MongoDB
        if self._mongo is not None:
            try:
                self._mongo.glas_med.documents.update_one(
                    {"id": doc_id},
                    {"$set": {**record, "vector": vec}},
                    upsert=True
                )
            except Exception as e:
                logger.warning("Failed writing document to MongoDB: %s", e)

        # Persist to Neo4j
        if self._graph is not None:
            try:
                with self._graph.session() as session:
                    session.run("MERGE (p:Paper {id: $id}) SET p += $record", id=doc_id, record=record)
            except Exception as e:
                logger.warning("Failed writing paper node to Neo4j: %s", e)

        return doc_id

    def search(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        query_vector = self._embedding(query)
        doc_ids_with_scores: list[tuple[str, float]] = []

        # 1. Primary path: Query FAISS Index
        if self._faiss_index is not None and self._faiss_index.ntotal > 0:
            import numpy as np
            q_mat = np.array([query_vector], dtype=np.float32)
            k = min(limit, self._faiss_index.ntotal)
            distances, indices = self._faiss_index.search(q_mat, k)
            
            for dist, idx in zip(distances[0], indices[0]):
                if idx in self._int_to_id:
                    doc_id = self._int_to_id[idx]
                    doc_ids_with_scores.append((doc_id, float(dist)))
        else:
            # Fallback path: In-memory dot product scan
            for doc_id, vec in self._vectors.items():
                score = sum(a * b for a, b in zip(query_vector, vec))
                doc_ids_with_scores.append((doc_id, float(score)))
            doc_ids_with_scores.sort(key=lambda x: x[1], reverse=True)
            doc_ids_with_scores = doc_ids_with_scores[:limit]

        if not doc_ids_with_scores:
            return []

        doc_ids = [did for did, _ in doc_ids_with_scores]
        score_map = dict(doc_ids_with_scores)

        # 2. Hydrate from MongoDB
        hydrated_docs: dict[str, dict[str, Any]] = {}
        if self._mongo is not None:
            try:
                mongo_records = list(self._mongo.glas_med.documents.find({"id": {"$in": doc_ids}}))
                for m_doc in mongo_records:
                    m_doc.pop("_id", None)
                    m_doc.pop("vector", None)
                    hydrated_docs[m_doc["id"]] = m_doc
            except Exception as e:
                logger.warning("MongoDB hydration error: %s", e)

        # 3. Fallback to in-memory store for any missing documents
        results = []
        for doc_id in doc_ids:
            doc = hydrated_docs.get(doc_id) or self._documents.get(doc_id)
            if doc:
                results.append({
                    **doc,
                    "similarity": round(score_map.get(doc_id, 0.0), 4)
                })

        return results

    def link_claim(self, claim: dict[str, Any]) -> None:
        if self._graph is None:
            return
        try:
            with self._graph.session() as session:
                session.run(
                    "MERGE (c:ClinicalClaim {id: $id}) SET c += $claim "
                    "WITH c MATCH (p:Paper {id: $paper_id}) MERGE (p)-[:REPORTS]->(c)",
                    id=claim["id"], paper_id=claim.get("paper_id", ""), claim=claim,
                )
        except Exception as e:
            logger.warning("Neo4j link_claim error: %s", e)
