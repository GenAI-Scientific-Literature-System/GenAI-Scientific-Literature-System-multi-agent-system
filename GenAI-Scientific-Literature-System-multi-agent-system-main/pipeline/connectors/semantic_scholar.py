# pipeline/connectors/semantic_scholar.py

import requests
import time
import os
from typing import Any
import threading

# Global rate limiter for Semantic Scholar (1 req/sec across all threads)
_SS_LOCK = threading.Lock()
_SS_LAST_CALL = 0.0

def _ss_rate_limit():
    global _SS_LAST_CALL
    with _SS_LOCK:
        now = time.time()
        if now - _SS_LAST_CALL < 1.0:
            time.sleep(1.0 - (now - _SS_LAST_CALL))
        _SS_LAST_CALL = time.time()

SEMANTIC_SCHOLAR_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

FIELDS = ",".join([
    "paperId",
    "title",
    "abstract",
    "year",
    "externalIds",
    "authors",
    "journal",
    "citationCount",
])


class SemanticScholarConnector:
    def __init__(self, top_k: int = 10, api_key: str | None = None, debug: bool = True):
        self.top_k = top_k
        self.api_key = api_key or os.getenv("SEMANTIC_SCHOLAR_API_KEY")
        self.debug = debug
        self.source = "semantic_scholar"

    def _headers(self) -> dict:
        headers = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        return headers

    def _get(self, url: str, params: dict[str, Any], timeout: int = 10):
        last_error: Exception | None = None
        for _ in range(2):
            try:
                _ss_rate_limit()
                return requests.get(url, params=params, headers=self._headers(), timeout=timeout)
            except Exception as e:
                last_error = e
                if self.debug:
                    print(f"[SemanticScholar] Request attempt failed: {e}")
                continue
        if last_error is not None:
            raise last_error
        raise RuntimeError("Request failed without an explicit exception")

    def _parse_paper(self, result: dict) -> dict[str, Any] | None:
        abstract = (result.get("abstract") or "").strip()
        if not abstract:
            return None

        title = (result.get("title") or "").strip()
        if not title:
            return None
        title = " ".join(title.split())

        abstract = " ".join(abstract.split())

        paper_id = result.get("paperId")
        if not paper_id:
            return None
        external_ids = result.get("externalIds") or {}
        doi = external_ids.get("DOI")
        doi = doi.lower().strip() if isinstance(doi, str) else None

        url = f"https://www.semanticscholar.org/paper/{paper_id}"

        year = result.get("year")
        try:
            year = int(year) if year else None
        except (ValueError, TypeError):
            year = None

        authors_raw = result.get("authors") or []
        authors = [a.get("name", "").strip() for a in authors_raw if a.get("name")]

        journal_raw = result.get("journal") or {}
        journal = journal_raw.get("name") if isinstance(journal_raw, dict) else None

        citation_count = result.get("citationCount")

        return {
            "paper_id": f"ss_{paper_id}",
            "title": title,
            "abstract": abstract,
            "source": self.source,
            "year": year,
            "doi": doi,
            "url": url,
            "journal": journal,
            "authors": authors,
            "citation_count": citation_count,  # useful for Agent 3 reliability scoring
            "score": None
        }

    def fetch(self, query: str) -> list[dict[str, Any]]:
        params = {
            "query": query,
            "limit": self.top_k,
            "fields": FIELDS,
        }

        try:
            response = self._get(SEMANTIC_SCHOLAR_SEARCH_URL, params=params)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            if self.debug:
                print(f"[SemanticScholar] Request failed: {e}")
            return []

        results = data.get("data", [])
        if self.debug:
            print(f"[SemanticScholar] Retrieved {len(results)} raw results")

        papers = []
        for result in results:
            parsed = self._parse_paper(result)
            if parsed:
                papers.append(parsed)

        papers = papers[:self.top_k]

        if self.debug:
            print(f"[SemanticScholar] Parsed {len(papers)} papers with abstracts")

        return papers