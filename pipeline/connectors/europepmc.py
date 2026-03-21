# pipeline/connectors/europepmc.py

import requests
import time
import hashlib
from typing import Any

EUROPEPMC_SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"


class EuropePMCConnector:
    def __init__(self, top_k: int = 10, debug: bool = True):
        self.top_k = top_k
        self.debug = debug
        self.source = "europepmc"

    def _get(self, url: str, params: dict[str, Any], timeout: int = 10):
        last_error: Exception | None = None
        for _ in range(2):
            try:
                time.sleep(0.1)
                return requests.get(url, params=params, timeout=timeout)
            except Exception as e:
                last_error = e
                continue
        if last_error is not None:
            raise last_error
        raise RuntimeError("Request failed without an explicit exception")

    def _parse_paper(self, result: dict) -> dict[str, Any] | None:
        abstract = result.get("abstractText", "").strip()
        if not abstract:
            return None  # skip papers with no abstract
        abstract = " ".join(abstract.split())

        pmid = result.get("pmid") or result.get("id", "")
        if pmid == "":
            pmid = None
        doi = result.get("doi")
        doi = doi.strip() if isinstance(doi, str) else None
        if doi == "":
            doi = None
        source_code = result.get("source")
        record_id = result.get("id")
        title = result.get("title", "").strip()
        if not title:
            return None
        year_raw = result.get("pubYear")
        authors_raw = result.get("authorString", "")
        journal = result.get("journalTitle")

        try:
            year = int(year_raw) if year_raw else None
        except ValueError:
            year = None

        authors = [a.strip() for a in authors_raw.split(",") if a.strip()]
        fallback_hash = hashlib.sha1(title.encode("utf-8")).hexdigest()[:16]
        identifier = pmid or doi or fallback_hash

        if isinstance(source_code, str) and source_code and isinstance(record_id, str) and record_id:
            url = f"https://europepmc.org/article/{source_code}/{record_id}"
        elif pmid:
            url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        elif doi:
            url = f"https://doi.org/{doi}"
        else:
            url = None

        # Extract open/free PDF URL if available
        pdf_url = None
        full_text_urls = result.get("fullTextUrlList", {}).get("fullTextUrl", [])
        for ft in full_text_urls:
            availability = (ft.get("availability") or "").strip().lower()
            doc_style = (ft.get("documentStyle") or "").strip().lower()
            if availability in {"free", "open access"} and doc_style == "pdf":
                pdf_url = ft.get("url")
                break

        return {
            "paper_id": f"europepmc_{identifier}",
            "title": title,
            "abstract": abstract,
            "source": self.source,
            "year": year,
            "doi": doi,
            "url": url,
            "pdf_url": pdf_url,
            "journal": journal,
            "authors": authors,
            "score": None
        }

    def fetch(self, query: str) -> list[dict[str, Any]]:
        params = {
            "query": query,
            "resultType": "core",      # includes abstract
            "pageSize": self.top_k,
            "format": "json",
        }

        try:
            response = self._get(EUROPEPMC_SEARCH_URL, params=params)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            if self.debug:
                print(f"[EuropePMC] Request failed: {e}")
            return []

        # Some parameter combinations can return a minimal payload.
        # Retry once with a simplified query form if no results structure is present.
        if "resultList" not in data:
            fallback_params = {
                "query": " ".join(query.split()),
                "resultType": "core",
                "pageSize": self.top_k,
                "format": "json",
            }
            try:
                response = self._get(EUROPEPMC_SEARCH_URL, params=fallback_params)
                response.raise_for_status()
                data = response.json()
            except Exception as e:
                if self.debug:
                    print(f"[EuropePMC] Fallback request failed: {e}")
                return []

        results = data.get("resultList", {}).get("result", [])
        if self.debug:
            print(f"[EuropePMC] Retrieved {len(results)} raw results")

        papers = []
        for result in results:
            parsed = self._parse_paper(result)
            if parsed:
                papers.append(parsed)

        papers = papers[:self.top_k]

        if self.debug:
            print(f"[EuropePMC] Parsed {len(papers)} papers with abstracts")

        return papers