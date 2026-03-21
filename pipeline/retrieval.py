# pipeline/retrieval.py

import os
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from typing import Any

from pipeline.connectors.pubmed import PubMedConnector
from pipeline.connectors.europepmc import EuropePMCConnector
from pipeline.connectors.semantic_scholar import SemanticScholarConnector
from pipeline.connectors.arxiv import ArxivConnector

# domain → sources mapping
DOMAIN_SOURCE_MAP: dict[str, list[str]] = {
    "medical":  ["pubmed", "europepmc"],
    "biology":  ["europepmc", "semantic_scholar"],
    "ml":       ["semantic_scholar", "arxiv"],
    "general":  ["semantic_scholar"],
}

MAX_SOURCES = 4


class Retriever:
    def __init__(
        self,
        top_k_per_source: int = 10,
        pubmed_api_key: str | None = None,
        semantic_scholar_api_key: str | None = None,
        debug: bool = True,
    ):
        self.top_k = top_k_per_source
        self.debug = debug

        # initialize all connectors once
        self.connectors: dict[str, Any] = {
            "pubmed": PubMedConnector(
                top_k=top_k_per_source,
                api_key=pubmed_api_key or os.getenv("PUBMED_API_KEY"),
                debug=debug,
            ),
            "europepmc": EuropePMCConnector(
                top_k=top_k_per_source,
                debug=debug,
            ),
            "semantic_scholar": SemanticScholarConnector(
                top_k=top_k_per_source,
                api_key=semantic_scholar_api_key or os.getenv("SEMANTIC_SCHOLAR_API_KEY"),
                debug=debug,
            ),
            "arxiv": ArxivConnector(
                top_k=top_k_per_source,
                debug=debug,
            ),
        }

    def _resolve_sources(self, domains: list[str]) -> list[str]:
        # map domains to sources, deduplicate, cap at MAX_SOURCES
        sources = []
        seen = set()
        for domain in domains:
            for source in DOMAIN_SOURCE_MAP.get(domain, ["semantic_scholar"]):
                if source not in seen:
                    sources.append(source)
                    seen.add(source)
        return sources[:MAX_SOURCES]

    def _fetch_from_source(self, source: str, query: str) -> tuple[str, list[dict[str, Any]]]:
        # returns (source_name, papers) — tuple so we know which source succeeded/failed
        try:
            connector = self.connectors[source]
            papers = connector.fetch(query)
            if self.debug:
                print(f"[Retriever] {source} → {len(papers)} papers")
            return source, papers
        except Exception as e:
            if self.debug:
                print(f"[Retriever] {source} failed: {e}")
            return source, []

    def _deduplicate(self, papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen = set()
        unique: list[dict[str, Any]] = []

        for p in papers:
            doi = p.get("doi")
            if isinstance(doi, str):
                doi = doi.lower().strip()
            key = doi or p.get("paper_id")

            if key and key not in seen:
                seen.add(key)
                unique.append(p)

        return unique

    def _rank(self, papers: list[dict[str, Any]], query: str) -> list[dict[str, Any]]:
        # Placeholder ranking. Keep stable order from dedup.
        _ = query
        return papers

    def retrieve(self, query: str, domains: list[str]) -> list[dict[str, Any]]:
        if not domains:
            domains = ["general"]

        sources = self._resolve_sources(domains)
        if not sources:
            sources = ["semantic_scholar"]

        if self.debug:
            print(f"[Retriever] Domains: {domains} → Sources: {sources}")

        all_papers: list[dict[str, Any]] = []

        # run all sources in parallel
        max_workers = min(4, len(sources))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._fetch_from_source, source, query): source
                for source in sources
            }

            for future in as_completed(futures):
                source_name = futures[future]
                try:
                    _, papers = future.result(timeout=10)
                except TimeoutError:
                    if self.debug:
                        print(f"[Retriever] {source_name} timed out")
                    continue
                except Exception as e:
                    if self.debug:
                        print(f"[Retriever] {source_name} failed: {e}")
                    continue
                all_papers.extend(papers)

        if self.debug:
            print(f"[Retriever] Total raw papers before dedup: {len(all_papers)}")

        all_papers = self._deduplicate(all_papers)

        if self.debug:
            print(f"[Retriever] Total papers after dedup: {len(all_papers)}")

        return self._rank(all_papers, query)