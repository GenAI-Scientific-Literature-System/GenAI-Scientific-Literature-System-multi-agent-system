# pipeline/aggregator.py

import re
import numpy as np
from datetime import datetime
from typing import Any
from pipeline.embedding import EmbeddingEngine

SIMILARITY_THRESHOLD = 0.92  # abstracts above this are considered duplicates


def _normalize_title(title: str) -> str:
    # lowercase, strip punctuation, collapse whitespace
    title = title.lower().strip()
    title = re.sub(r"[^\w\s]", "", title)
    title = re.sub(r"\s+", " ", title)
    return title


class Aggregator:
    def __init__(
        self,
        embedding_engine: EmbeddingEngine,
        similarity_threshold: float = SIMILARITY_THRESHOLD,
        final_top_k: int = 20,
        debug: bool = True,
    ):
        self.engine = embedding_engine
        self.threshold = similarity_threshold
        self.final_top_k = final_top_k
        self.debug = debug

    def _dedup_by_doi(self, papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen_dois: set[str] = set()
        unique: list[dict[str, Any]] = []

        for paper in papers:
            doi = paper.get("doi")
            if isinstance(doi, str) and doi:
                if doi in seen_dois:
                    if self.debug:
                        print(f"[Aggregator] DOI dedup: {paper.get('paper_id')}")
                    continue
                seen_dois.add(doi)
            unique.append(paper)

        if self.debug:
            print(f"[Aggregator] After DOI dedup: {len(unique)} papers")
        return unique

    def _dedup_by_title(self, papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen_titles: set[str] = set()
        unique: list[dict[str, Any]] = []

        for paper in papers:
            title = paper.get("title", "")
            normalized = _normalize_title(title)
            if not normalized:
                unique.append(paper)
                continue
            if normalized in seen_titles:
                if self.debug:
                    print(f"[Aggregator] Title dedup: {paper.get('paper_id')}")
                continue
            seen_titles.add(normalized)
            unique.append(paper)

        if self.debug:
            print(f"[Aggregator] After title dedup: {len(unique)} papers")
        return unique

    def _dedup_by_embedding(self, papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
        unique: list[dict[str, Any]] = []

        for i, paper in enumerate(papers):
            emb_i = paper.get("embedding")
            if emb_i is None:
                unique.append(paper)
                continue

            is_duplicate = False
            for kept in unique:
                emb_k = kept.get("embedding")
                if emb_k is None:
                    continue
                sim = self.engine.compute_similarity(emb_i, emb_k)
                if sim >= self.threshold:
                    if self.debug:
                        print(
                            f"[Aggregator] Embedding dedup: {paper.get('paper_id')} "
                            f"~ {kept.get('paper_id')} (sim={sim:.3f})"
                        )
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique.append(paper)

        if self.debug:
            print(f"[Aggregator] After embedding dedup: {len(unique)} papers")
        return unique

    def _boost_score(self, paper: dict[str, Any]) -> float:
        # base score from embedding similarity to query
        score = paper.get("score") or 0.0

        # recency boost only — citation count removed to avoid bias
        year = paper.get("year")
        if year and isinstance(year, int):
            current_year = datetime.now().year
            age = current_year - year
            if age <= 5:
                score += 0.02 * (5 - age)

        return score

    def aggregate(
        self,
        papers: list[dict[str, Any]],
        query: str,
    ) -> list[dict[str, Any]]:
        if not papers:
            return []

        if self.debug:
            print(f"[Aggregator] Starting with {len(papers)} raw papers")

        # layer 1 — DOI dedup (cheapest, do first)
        papers = self._dedup_by_doi(papers)

        # layer 2 — title dedup
        papers = self._dedup_by_title(papers)

        # embed + score (keep embeddings for layer 3)
        papers = self.engine.process(papers, query, keep_embeddings=True)

        # layer 3 — embedding dedup
        papers = self._dedup_by_embedding(papers)

        # apply boosted scoring
        for paper in papers:
            paper["score"] = self._boost_score(paper)

        # dynamic threshold based on score distribution
        scores = [p.get("score") or 0.0 for p in papers]
        mean = float(np.mean(scores))
        std = float(np.std(scores))
        threshold = mean - 0.5 * std
        papers = [p for p in papers if (p.get("score") or 0.0) >= threshold]
        if self.debug:
            print(f"[Aggregator] Dynamic threshold: {threshold:.4f} (mean={mean:.4f}, std={std:.4f})")

        # sort by final boosted score
        papers = sorted(papers, key=lambda p: p.get("score", 0.0), reverse=True)

        # cap to final_top_k
        papers = papers[:self.final_top_k]

        if self.debug:
            print(f"[Aggregator] Final output: {len(papers)} papers")

        return papers