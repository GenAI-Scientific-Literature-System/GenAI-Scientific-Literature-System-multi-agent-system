import os
import re
import sqlite3
import json
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from typing import Any
import numpy as np

from pipeline.connectors.pubmed import PubMedConnector
from pipeline.connectors.europepmc import EuropePMCConnector
from pipeline.connectors.semantic_scholar import SemanticScholarConnector
from pipeline.connectors.arxiv import ArxivConnector
from pipeline.embedding import EmbeddingEngine

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
os.makedirs(CACHE_DIR, exist_ok=True)
CACHE_DB = os.path.join(CACHE_DIR, "retrieval_cache.db")

def _init_cache_db():
    try:
        with sqlite3.connect(CACHE_DB) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS query_cache (
                    cache_key TEXT PRIMARY KEY,
                    papers_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
    except Exception:
        pass

_init_cache_db()

# domain → sources mapping
DOMAIN_SOURCE_MAP: dict[str, list[str]] = {
    "medical":  ["pubmed", "europepmc"],
    "biology":  ["europepmc", "semantic_scholar"],
    "ml":       ["semantic_scholar", "arxiv"],
    "general":  ["semantic_scholar"],
}

MAX_SOURCES = 4

STOP_TERMS = {
    "the", "and", "for", "with", "that", "this", "from", "into", "than", "does",
    "what", "when", "where", "which", "while", "have", "has", "had", "are", "was",
    "were", "been", "being", "about", "across", "using", "use", "used", "study",
    "studies", "paper", "papers", "system", "systems", "model", "models",
}


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
        
        # Initialize embedding engine for semantic domain coherence filtering
        try:
            self.embedding_engine = EmbeddingEngine(debug=debug)
        except Exception as e:
            if debug:
                print(f"[Retriever] Failed to initialize EmbeddingEngine: {e}")
            self.embedding_engine = None

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

    def _source_queries(self, source: str, query: str, domains: list[str]) -> list[str]:
        profile = self._query_profile(query, domains)
        terms = profile["ordered_terms"][:6]
        phrases = [phrase for phrase in profile["phrases"][:4] if len(phrase.split()) >= 2]
        compact_terms = " ".join(terms)
        queries: list[str] = []

        if source == "semantic_scholar":
            return [query]

        if source in {"pubmed", "europepmc"}:
            queries.append(query)
            if compact_terms:
                queries.append(compact_terms)
            if phrases:
                queries.append(" ".join(f"\"{phrase}\"" for phrase in phrases[:2]))
            return list(dict.fromkeys(q.strip() for q in queries if q.strip()))

        if source == "arxiv":
            queries.append(query)
            if compact_terms:
                queries.append(compact_terms)
            return list(dict.fromkeys(q.strip() for q in queries if q.strip()))

        return [query]

    def _fetch_from_source(self, source: str, query: str, domains: list[str]) -> tuple[str, list[dict[str, Any]]]:
        # returns (source_name, papers) — tuple so we know which source succeeded/failed
        try:
            connector = self.connectors[source]
            variants = self._source_queries(source, query, domains)
            merged: list[dict[str, Any]] = []
            seen_keys: set[str] = set()
            for source_query in variants:
                papers = connector.fetch(source_query)
                for paper in papers:
                    key = str((paper.get("doi") or paper.get("paper_id") or paper.get("title") or "")).lower().strip()
                    if not key or key in seen_keys:
                        continue
                    seen_keys.add(key)
                    merged.append(paper)
            if self.debug:
                print(f"[Retriever] {source} {variants} → {len(merged)} papers")
            return source, merged
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

    def _tokenize(self, text: str) -> list[str]:
        return [
            token for token in re.findall(r"[a-zA-Z0-9]+", (text or "").lower())
            if len(token) >= 3 and token not in STOP_TERMS
        ]

    def _query_profile(self, query: str, domains: list[str]) -> dict[str, Any]:
        lowered = (query or "").lower()
        terms = self._tokenize(query)
        ordered_terms = list(dict.fromkeys(terms))
        weighted_terms = [
            (term, 1.0 + min(1.5, (len(term) - 3) * 0.18))
            for term in terms
        ]
        bigrams = [" ".join(pair) for pair in zip(terms, terms[1:])]
        trigrams = [" ".join(triple) for triple in zip(terms, terms[1:], terms[2:])]
        phrase_candidates: list[str] = []
        if trigrams:
            phrase_candidates.append(trigrams[0])
            if len(trigrams) > 1:
                phrase_candidates.append(trigrams[-1])
        if bigrams:
            phrase_candidates.append(bigrams[0])
            if len(bigrams) > 1:
                phrase_candidates.append(bigrams[-1])
        phrases = [phrase for phrase in dict.fromkeys(phrase_candidates) if phrase]

        salient_terms = [term for term, _ in sorted(weighted_terms, key=lambda item: (-item[1], item[0]))[:8]]
        title_probe_terms = salient_terms[:4]
        dense_query = " ".join(terms)

        return {
            "terms": terms,
            "ordered_terms": ordered_terms,
            "weighted_terms": weighted_terms,
            "phrases": phrases,
            "salient_terms": salient_terms,
            "title_probe_terms": title_probe_terms,
            "dense_query": dense_query,
            "lowered_query": lowered,
            "domains": domains,
        }

    def _paper_relevance(self, paper: dict[str, Any], profile: dict[str, Any]) -> float:
        title = str(paper.get("title") or "").lower()
        abstract = str(paper.get("abstract") or "").lower()
        categories = " ".join(paper.get("categories") or []).lower()
        journal = str(paper.get("journal") or "").lower()
        haystack = " ".join(part for part in (title, abstract, categories, journal) if part)

        score = 0.0
        weighted_terms: list[tuple[str, float]] = profile["weighted_terms"]
        matched_weight = 0.0
        title_matched_weight = 0.0
        total_weight = sum(weight for _, weight in weighted_terms) or 1.0

        for term, weight in weighted_terms:
            if term in title:
                matched_weight += weight
                title_matched_weight += weight
                score += 2.2 * weight
            elif term in abstract:
                matched_weight += weight
                score += 0.85 * weight

        coverage = matched_weight / total_weight
        title_coverage = title_matched_weight / total_weight

        phrase_hits = 0
        title_phrase_hits = 0
        for phrase in profile["phrases"]:
            if phrase in title:
                phrase_hits += 1
                title_phrase_hits += 1
                score += 2.4
            elif phrase in abstract:
                phrase_hits += 1
                score += 1.1

        salient_terms = profile["salient_terms"]
        salient_hit_count = sum(1 for term in salient_terms if term in haystack)
        title_probe_terms = profile["title_probe_terms"]
        title_probe_hits = sum(1 for term in title_probe_terms if term in title)

        score += coverage * 8.0
        score += title_coverage * 5.0
        score += min(3.0, salient_hit_count * 0.7)
        score += min(1.5, title_probe_hits * 0.5)

        if profile["dense_query"] and profile["dense_query"] in haystack:
            score += 4.0

        if coverage < 0.22:
            score -= (0.22 - coverage) * 18.0
        if title_coverage < 0.08:
            score -= (0.08 - title_coverage) * 12.0
        if phrase_hits == 0 and coverage < 0.35:
            score -= 2.5
        if title_probe_hits == 0 and coverage < 0.45:
            score -= 1.5

        source = str(paper.get("source") or "")
        if "medical" in profile["domains"]:
            if source in {"pubmed", "europepmc"}:
                score += 0.5
            elif source == "arxiv":
                score -= 0.4
        elif "ml" in profile["domains"] and source in {"semantic_scholar", "arxiv"}:
            score += 0.25

        year = paper.get("year")
        if isinstance(year, int):
            score += max(0, min(0.8, (year - 2018) * 0.04))

        paper["query_coverage"] = round(coverage, 4)
        paper["title_query_coverage"] = round(title_coverage, 4)
        paper["phrase_hits"] = phrase_hits
        paper["title_phrase_hits"] = title_phrase_hits

        return score

    def _rank(self, papers: list[dict[str, Any]], query: str) -> list[dict[str, Any]]:
        return self._rank_with_domains(papers, query, ["general"])

    def _get_domain_reference_texts(self, domains: list[str]) -> list[str]:
        """Return semantic reference texts for each domain."""
        domain_refs = {
            "medical": [
                "biomedical research clinical medicine healthcare disease treatment patient",
                "pharmaceutical therapeutic drug diagnosis health hospital medical science",
                "disease pathology symptoms clinical trial medical condition health",
            ],
            "biology": [
                "biological research molecular biology genetics cell tissue organism species",
                "biological system evolution ecology life science bioscience organism",
                "protein gene mutation cellular biological pathway organism",
            ],
            "ml": [
                "machine learning artificial intelligence deep learning neural network algorithm",
                "model training data learning computational classification prediction",
            ],
        }
        refs = []
        for domain in domains:
            refs.extend(domain_refs.get(domain, []))
        return refs if refs else ["scientific research paper"]

    def _compute_domain_coherence(self, paper: dict[str, Any], domains: list[str]) -> float:
        """Compute semantic similarity between paper and domain using embeddings."""
        try:
            # Get paper text
            abstract = str(paper.get("abstract") or "").strip()
            title = str(paper.get("title") or "").strip()
            if not abstract and not title:
                return 0.0
            
            paper_text = f"{title} {abstract}"[:512]  # Limit to avoid too-long texts
            
            # Get domain reference texts and compute their mean embedding
            ref_texts = self._get_domain_reference_texts(domains)
            if not ref_texts:
                return 1.0  # No domain filter if no references
            
            # Embed paper and references
            paper_embedding = self.embedding_engine.embed_query(paper_text)
            ref_embeddings = self.embedding_engine._embed(ref_texts)
            
            # Compute mean similarity to domain references
            similarities = np.dot(ref_embeddings, paper_embedding)  # Cosine similarities
            mean_similarity = float(np.mean(similarities))
            return mean_similarity
        except Exception as e:
            if self.debug:
                print(f"[Retriever] Domain coherence computation failed: {e}")
            return 0.5  # Neutral score on error

    def _rank_with_domains(self, papers: list[dict[str, Any]], query: str, domains: list[str]) -> list[dict[str, Any]]:
        profile = self._query_profile(query, domains)
        if not profile["terms"] and not profile["phrases"]:
            return papers

        scored: list[tuple[float, int, dict[str, Any]]] = []
        for idx, paper in enumerate(papers):
            score = self._paper_relevance(paper, profile)
            paper["retrieval_score"] = score
            scored.append((score, idx, paper))

        scored.sort(key=lambda row: (-row[0], row[1]))
        ranked = [row[2] for row in scored]
        if not ranked:
            return ranked

        # Semantic domain coherence filter: remove papers off-topic for specific domains
        if ("medical" in domains or "biology" in domains) and self.embedding_engine:
            coherence_threshold = 0.60  # Papers below this semantic similarity to domain are filtered
            filtered_ranked = []
            for paper in ranked:
                coherence = self._compute_domain_coherence(paper, [d for d in domains if d in {"medical", "biology"}])
                paper["domain_coherence"] = round(coherence, 4)
                if coherence >= coherence_threshold:
                    filtered_ranked.append(paper)
            if filtered_ranked:
                ranked = filtered_ranked

        best_score = scored[0][0]
        best_coverage = max(float(p.get("query_coverage") or 0.0) for p in ranked)
        best_title_coverage = max(float(p.get("title_query_coverage") or 0.0) for p in ranked)
        score_floor = max(1.0, best_score * 0.35)
        coverage_floor = max(0.12, best_coverage * 0.35)
        title_floor = max(0.04, best_title_coverage * 0.35)

        filtered = [
            paper for paper in ranked
            if float(paper.get("retrieval_score") or 0.0) >= score_floor
            and (
                float(paper.get("query_coverage") or 0.0) >= coverage_floor
                or float(paper.get("title_query_coverage") or 0.0) >= title_floor
                or int(paper.get("phrase_hits") or 0) > 0
            )
        ]

        if filtered:
            return filtered[:self.top_k * 2]

        fallback = [
            paper for paper in ranked
            if float(paper.get("retrieval_score") or 0.0) > 0.0
        ]
        return fallback[:min(self.top_k * 3, len(fallback))] if fallback else ranked[:min(5, len(ranked))]

    def retrieve(self, query: str, domains: list[str]) -> list[dict[str, Any]]:
        if not domains:
            domains = ["general"]

        cache_key = f"{query.lower().strip()}_{sorted(domains)}_{self.top_k}"
        try:
            with sqlite3.connect(CACHE_DB) as conn:
                cur = conn.cursor()
                cur.execute("SELECT papers_json FROM query_cache WHERE cache_key = ?", (cache_key,))
                row = cur.fetchone()
                if row and row[0]:
                    cached_papers = json.loads(row[0])
                    if self.debug:
                        print(f"[Retriever] SQLite Disk Cache HIT for '{query}' ({len(cached_papers)} papers)")
                    return cached_papers
        except Exception:
            pass

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
                executor.submit(self._fetch_from_source, source, query, domains): source
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

        results = self._rank_with_domains(all_papers, query, domains)
        try:
            with sqlite3.connect(CACHE_DB) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO query_cache (cache_key, papers_json) VALUES (?, ?)",
                    (cache_key, json.dumps(results)),
                )
        except Exception:
            pass
        return results
