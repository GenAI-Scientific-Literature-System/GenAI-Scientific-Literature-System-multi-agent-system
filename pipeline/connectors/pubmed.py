# pipeline/connectors/pubmed.py

import requests
import os
import time
from datetime import datetime
from typing import Any

PUBMED_SEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_FETCH_URL  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
PUBMED_SUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"

CURRENT_YEAR = datetime.now().year


class PubMedConnector:
    def __init__(self, top_k: int = 10, api_key: str | None = None, debug: bool = True):
        self.top_k = top_k
        self.api_key = api_key or os.getenv("PUBMED_API_KEY")
        self.debug = debug
        self.source = "pubmed"

    def _base_params(self) -> dict:
        params = {}
        if self.api_key:
            params["api_key"] = self.api_key
        return params

    def _get(self, url: str, params: dict[str, Any], timeout: int):
        last_error: Exception | None = None
        for _ in range(2):
            try:
                # Lightweight pacing to reduce API burst issues.
                time.sleep(0.1)
                return requests.get(url, params=params, timeout=timeout)
            except Exception as e:
                last_error = e
                continue
        if last_error is not None:
            raise last_error
        raise RuntimeError("Request failed without an explicit exception")

    def _get_text(self, element) -> str:
        return "".join(element.itertext()).strip() if element is not None else ""

    def _extract_year(self, article) -> int | None:
        year_el = article.find(".//PubDate/Year")
        if year_el is not None and year_el.text:
            try:
                return int(year_el.text)
            except ValueError:
                pass

        medline = article.find(".//PubDate/MedlineDate")
        if medline is not None and medline.text and len(medline.text) >= 4:
            try:
                return int(medline.text[:4])
            except ValueError:
                pass

        article_date_year = article.find(".//ArticleDate/Year")
        if article_date_year is not None and article_date_year.text:
            try:
                return int(article_date_year.text)
            except ValueError:
                pass

        return None

    def _search(self, query: str) -> list[str]:
        # returns list of PubMed IDs (PMIDs)
        params = self._base_params()
        params.update({
            "db": "pubmed",
            "term": query,
            "retmax": self.top_k,
            "retmode": "json",
            "sort": "relevance",
        })

        try:
            response = self._get(PUBMED_SEARCH_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            pmids = data.get("esearchresult", {}).get("idlist", [])
            if self.debug:
                print(f"[PubMed] Found {len(pmids)} PMIDs")
            return pmids
        except Exception as e:
            if self.debug:
                print(f"[PubMed] Search failed: {e}")
            return []

    def _fetch(self, pmids: list[str]) -> list[dict[str, Any]]:
        # fetches abstracts + metadata for given PMIDs
        if not pmids:
            return []

        params = self._base_params()
        params.update({
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
            "rettype": "abstract",
        })

        try:
            response = self._get(PUBMED_FETCH_URL, params=params, timeout=15)
            response.raise_for_status()
            return self._parse_xml(response.text)
        except Exception as e:
            if self.debug:
                print(f"[PubMed] Fetch failed: {e}")
            return []

    def _parse_xml(self, xml_text: str) -> list[dict[str, Any]]:
        # lightweight XML parsing without external deps
        import xml.etree.ElementTree as ET

        papers = []
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as e:
            if self.debug:
                print(f"[PubMed] XML parse error: {e}")
            return []

        for article in root.findall(".//PubmedArticle"):
            try:
                # title
                title_el = article.find(".//ArticleTitle")
                title = self._get_text(title_el)

                # abstract
                abstract_parts = article.findall(".//AbstractText")
                abstract = " ".join(
                    self._get_text(el) for el in abstract_parts
                ).strip()

                # DOI
                doi = None
                for id_el in article.findall(".//ArticleId"):
                    if id_el.get("IdType") == "doi":
                        doi = id_el.text
                        break

                # PMID
                pmid_el = article.find(".//PMID")
                pmid = pmid_el.text if pmid_el is not None else None

                # URL
                if pmid:
                    url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                elif doi:
                    url = f"https://doi.org/{doi}"
                else:
                    url = None

                # year
                year = self._extract_year(article)

                # authors
                authors = []
                for author in article.findall(".//Author"):
                    last = author.find("LastName")
                    fore = author.find("ForeName")
                    if last is not None:
                        name = last.text
                        if fore is not None:
                            name += f", {fore.text}"
                        authors.append(name)

                if not abstract:
                    continue  # skip papers with no abstract

                papers.append({
                    "paper_id": f"pubmed_{pmid}",
                    "title": title,
                    "abstract": abstract,
                    "source": self.source,
                    "year": year,
                    "doi": doi,
                    "url": url,
                    "authors": authors,
                    "score": None
                })

            except Exception as e:
                if self.debug:
                    print(f"[PubMed] Failed to parse article: {e}")
                continue

        if self.debug:
            print(f"[PubMed] Parsed {len(papers)} papers with abstracts")

        return papers

    def fetch(self, query: str) -> list[dict[str, Any]]:
        pmids = self._search(query)
        if not pmids:
            return []
        return self._fetch(pmids)