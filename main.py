# main.py

import os
import json
from dotenv import load_dotenv

from utils.api_keys import load_groq_api_keys
from utils.model_config import DEFAULT_CLAIM_EXTRACTION_MODEL, DEFAULT_EVIDENCE_COLLECTION_MODEL

from pipeline.preprocessing import QueryPreprocessor
from pipeline.retrieval import Retriever
from pipeline.embedding import EmbeddingEngine
from pipeline.aggregator import Aggregator
from pipeline.clustering import PaperClusterer

from agents.claim_extraction import ClaimExtractor
from agents.evidence_collection import EvidenceCollector

load_dotenv()


def run_pipeline(
    query: str,
    top_k_per_source: int = 10,
    final_top_k: int = 15,
    debug: bool = True,
) -> dict:

    api_keys = load_groq_api_keys()

    # ── Step 1: Preprocessing ─────────────────────────────────────────────
    if debug:
        print("\n=== STEP 1: Query Preprocessing ===")
    preprocessor = QueryPreprocessor(api_key=api_keys, debug=debug)
    processed = preprocessor.process(query)
    cleaned_query = processed["cleaned_query"]
    domains = processed["domains"]
    if debug:
        print(f"Cleaned query : {cleaned_query}")
        print(f"Domains       : {domains}")

    # ── Step 2: Retrieval ─────────────────────────────────────────────────
    if debug:
        print("\n=== STEP 2: Retrieval ===")
    retriever = Retriever(top_k_per_source=top_k_per_source, debug=debug)
    raw_papers = retriever.retrieve(query=cleaned_query, domains=domains)
    if debug:
        print(f"Raw papers retrieved: {len(raw_papers)}")

    if not raw_papers:
        print("[main] No papers retrieved. Exiting.")
        return {"query": query, "papers": [], "results": []}

    # ── Step 3: Embedding + Aggregation ───────────────────────────────────
    if debug:
        print("\n=== STEP 3: Embedding + Aggregation ===")
    engine = EmbeddingEngine(debug=debug)
    aggregator = Aggregator(
        embedding_engine=engine,
        final_top_k=final_top_k,
        debug=debug
    )

    # keep embeddings for clustering
    raw_papers = engine.process(raw_papers, cleaned_query, keep_embeddings=True)
    final_papers = aggregator.aggregate(raw_papers, query=cleaned_query)

    if debug:
        print(f"Papers after aggregation: {len(final_papers)}")

    if not final_papers:
        print("[main] No papers passed aggregation threshold. Exiting.")
        return {"query": query, "papers": [], "results": []}

    # ── Step 4: Clustering ────────────────────────────────────────────────
    if debug:
        print("\n=== STEP 4: Clustering ===")
    clusterer = PaperClusterer(embedding_engine=engine, debug=debug)
    final_papers = clusterer.cluster(final_papers)

    # strip embeddings — agents don't need them
    for p in final_papers:
        p.pop("embedding", None)

    # ── Step 5: Claim Extraction (Agent 1) ────────────────────────────────
    if debug:
        print("\n=== STEP 5: Claim Extraction (Agent 1) ===")
    claim_extractor = ClaimExtractor(api_key=api_keys, model=DEFAULT_CLAIM_EXTRACTION_MODEL)

    for paper in final_papers:
        claim_result = claim_extractor.extract(paper.get("abstract", ""))
        paper["extracted_claim"] = claim_result.get("claim")
        paper["claim_confidence"] = claim_result.get("confidence")
        paper["claim_reasoning"] = claim_result.get("reasoning")
        if debug:
            print(f"  [{paper.get('paper_id')}] {paper.get('extracted_claim', 'FAILED')[:80]}")

    # ── Step 6: Evidence Collection (Agent 2) ─────────────────────────────
    if debug:
        print("\n=== STEP 6: Evidence Collection (Agent 2) ===")

    results = []
    collector = EvidenceCollector(api_key=api_keys, model=DEFAULT_EVIDENCE_COLLECTION_MODEL)

    for i, focal_paper in enumerate(final_papers):
        claim = focal_paper.get("extracted_claim")
        if not claim:
            continue

        # compare focal paper's claim against all other papers
        other_papers = [
            {
                "paper_id": p.get("paper_id"),
                "abstract": p.get("abstract", "")
            }
            for j, p in enumerate(final_papers) if j != i
        ]

        if not other_papers:
            continue

        evidence = collector.collect(claim=claim, papers=other_papers)

        results.append({
            "focal_paper_id": focal_paper.get("paper_id"),
            "focal_paper_title": focal_paper.get("title"),
            "claim": claim,
            "claim_confidence": focal_paper.get("claim_confidence"),
            "cluster_id": focal_paper.get("cluster_id"),
            "cluster_label": focal_paper.get("cluster_label"),
            "supporting": evidence.get("supporting", []),
            "contradicting": evidence.get("contradicting", []),
            "inconclusive": evidence.get("inconclusive", []),
        })

        if debug:
            print(
                f"  [{focal_paper.get('paper_id')}] "
                f"support={len(evidence.get('supporting', []))} "
                f"contradict={len(evidence.get('contradicting', []))} "
                f"inconclusive={len(evidence.get('inconclusive', []))}"
            )

    # ── Final Output ──────────────────────────────────────────────────────
    output = {
        "query": query,
        "cleaned_query": cleaned_query,
        "domains": domains,
        "papers": final_papers,
        "results": results,
    }

    if debug:
        print(f"\n=== PIPELINE COMPLETE ===")
        print(f"Papers analysed : {len(final_papers)}")
        print(f"Claims extracted: {len([p for p in final_papers if p.get('extracted_claim')])}")
        print(f"Evidence results: {len(results)}")

    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Multi-Agent Scientific Literature Analysis")
    parser.add_argument("query", type=str, help="Research query to analyse")
    parser.add_argument("--top-k", type=int, default=10, help="Papers per source (default: 10)")
    parser.add_argument("--final-k", type=int, default=15, help="Final papers after aggregation (default: 15)")
    parser.add_argument("--output", type=str, default=None, help="Save output to JSON file")
    parser.add_argument("--quiet", action="store_true", help="Suppress debug output")
    args = parser.parse_args()

    output = run_pipeline(
        query=args.query,
        top_k_per_source=args.top_k,
        final_top_k=args.final_k,
        debug=not args.quiet,
    )

    if args.output:
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\nOutput saved to {args.output}")
    else:
        print("\n--- Final JSON Output ---")
        print(json.dumps(output, indent=2, default=str))