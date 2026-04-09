# pipeline/embedding.py

import numpy as np
from typing import Any
from sentence_transformers import SentenceTransformer

SPECTER_MODEL = "allenai-specter"


class EmbeddingEngine:
    def __init__(self, model_name: str = SPECTER_MODEL, debug: bool = True):
        self.debug = debug
        if self.debug:
            print(f"[EmbeddingEngine] Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)
        if self.debug:
            print(f"[EmbeddingEngine] Model loaded")

    def _embed(self, texts: list[str]) -> np.ndarray:
        # batch encode all texts at once — much faster than one by one
        return self.model.encode(
            texts,
            batch_size=16,
            show_progress_bar=self.debug,
            convert_to_numpy=True,
            normalize_embeddings=True,  # L2 normalize → cosine sim becomes dot product
        )

    def embed_query(self, query: str) -> np.ndarray:
        return self._embed([query])[0]

    def embed_papers(self, papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
        # extract abstracts, embed, attach back to papers
        # only embed papers with non-empty abstracts to avoid waste
        valid_pairs = [(i, p.get("abstract", "") or "") for i, p in enumerate(papers) if p.get("abstract")]

        if not valid_pairs:
            return papers

        if self.debug:
            print(f"[EmbeddingEngine] Embedding {len(valid_pairs)} abstracts")

        indices, abstracts = zip(*valid_pairs)
        embeddings = self._embed(list(abstracts))

        for idx, embedding in zip(indices, embeddings):
            papers[idx]["embedding"] = embedding

        return papers

    def score_papers(
        self,
        papers: list[dict[str, Any]],
        query_embedding: np.ndarray
    ) -> list[dict[str, Any]]:
        # cosine similarity = dot product since embeddings are L2 normalized
        for paper in papers:
            embedding = paper.get("embedding")
            if embedding is not None:
                paper["score"] = float(np.dot(query_embedding, embedding))
            else:
                if self.debug:
                    print(f"[EmbeddingEngine] Missing embedding for paper {paper.get('paper_id')}")
                paper["score"] = 0.0
        return papers

    def compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        # used by aggregator for abstract-level dedup
        return float(np.dot(embedding1, embedding2))

    def process(
        self,
        papers: list[dict[str, Any]],
        query: str,
        keep_embeddings: bool = False
    ) -> list[dict[str, Any]]:
        # single entry point — embeds papers + query, scores, returns sorted
        query_embedding = self.embed_query(query)
        papers = self.embed_papers(papers)
        papers = self.score_papers(papers, query_embedding)
        papers = sorted(papers, key=lambda p: p.get("score", 0.0), reverse=True)
        
        if not keep_embeddings:
            # cleanup embeddings to save memory after scoring
            for p in papers:
                p.pop("embedding", None)
        
        return papers