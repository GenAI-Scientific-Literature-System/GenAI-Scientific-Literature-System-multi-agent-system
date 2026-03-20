# pipeline/connectors/arxiv.py

import requests
import time
import hashlib
import xml.etree.ElementTree as ET
from typing import Any

ARXIV_SEARCH_URL = "http://export.arxiv.org/api/query"

# XML namespaces used by arXiv Atom feed
ATOM_NS = "http://www.w3.org/2005/Atom"
ARXIV_NS = "http://arxiv.org/schemas/atom"


class ArxivConnector:
    def __init__(self, top_k: int = 10, debug: bool = True):
        self.top_k = top_k
        self.debug = debug
        self.source = "arxiv"

    def _get(self, url: str, params: dict[str, Any], timeout: int = 15):
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

    def _parse_feed(self, xml_text: str) -> list[dict[str, Any]]:
        papers = []

        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as e:
            if self.debug:
                print(f"[arXiv] XML parse error: {e}")
            return []

        for entry in root.findall(f"{{{ATOM_NS}}}entry"):
            try:
                title_el = entry.find(f"{{{ATOM_NS}}}title")
                title = " ".join((title_el.text or "").split()).strip() if title_el is not None else ""
                if not title:
                    continue

                abstract_el = entry.find(f"{{{ATOM_NS}}}summary")
                abstract = " ".join((abstract_el.text or "").split()).strip() if abstract_el is not None else ""
                if not abstract:
                    continue

                # arXiv ID from the <id> tag — looks like:
                # http://arxiv.org/abs/2301.00001v1
                id_el = entry.find(f"{{{ATOM_NS}}}id")
                raw_id = (id_el.text or "").strip() if id_el is not None else ""
                if not raw_id:
                    continue
                arxiv_id = raw_id.split("/abs/")[-1] if "/abs/" in raw_id else raw_id
                arxiv_id = arxiv_id.split("v")[0]

                # DOI — arXiv papers may have a published DOI
                doi = None
                doi_el = entry.find(f"{{{ARXIV_NS}}}doi")
                if doi_el is not None and doi_el.text:
                    doi = doi_el.text.lower().strip()

                # year from <published>
                published_el = entry.find(f"{{{ATOM_NS}}}published")
                year = None
                if published_el is not None and published_el.text:
                    try:
                        year = int(published_el.text[:4])
                    except ValueError:
                        pass

                # authors
                authors = []
                for author_el in entry.findall(f"{{{ATOM_NS}}}author"):
                    name_el = author_el.find(f"{{{ATOM_NS}}}name")
                    if name_el is not None and name_el.text:
                        authors.append(name_el.text.strip())

                # URL to paper
                url = f"https://arxiv.org/abs/{arxiv_id}"

                # Optional direct PDF link from Atom entry links
                pdf_url = None
                for link in entry.findall(f"{{{ATOM_NS}}}link"):
                    if link.get("title") == "pdf":
                        pdf_url = link.get("href")
                        break

                # categories e.g. cs.LG, cs.AI
                categories = []
                for cat_el in entry.findall(f"{{{ATOM_NS}}}category"):
                    term = cat_el.get("term")
                    if term:
                        categories.append(term)

                fallback_hash = hashlib.sha1(title.encode("utf-8")).hexdigest()[:16]
                identifier = arxiv_id or fallback_hash

                papers.append({
                    "paper_id": f"arxiv_{identifier}",
                    "title": title,
                    "abstract": abstract,
                    "source": self.source,
                    "year": year,
                    "doi": doi,
                    "url": url,
                    "pdf_url": pdf_url,
                    "journal": None,  # arXiv is a preprint server, no journal
                    "authors": authors,
                    "categories": categories,  # useful for domain filtering later
                    "citation_count": None,    # arXiv doesn't provide this
                    "score": None
                })

            except Exception as e:
                if self.debug:
                    print(f"[arXiv] Failed to parse entry: {e}")
                continue

        return papers

    def fetch(self, query: str) -> list[dict[str, Any]]:
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": self.top_k,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }

        try:
            response = self._get(ARXIV_SEARCH_URL, params=params)
            response.raise_for_status()
        except Exception as e:
            if self.debug:
                print(f"[arXiv] Request failed: {e}")
            return []

        papers = self._parse_feed(response.text)
        papers = papers[:self.top_k]

        if self.debug:
            print(f"[arXiv] Parsed {len(papers)} papers with abstracts")

        return papers