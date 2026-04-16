import os
import json
import argparse
from dotenv import load_dotenv

from utils.api_keys import load_groq_api_keys
from utils.model_config import (
    DEFAULT_CLAIM_EXTRACTION_MODEL,
    DEFAULT_EVIDENCE_COLLECTION_MODEL,
)

from pipeline.preprocessing import QueryPreprocessor
from pipeline.retrieval import Retriever
from pipeline.embedding import EmbeddingEngine
from pipeline.aggregator import Aggregator
from pipeline.clustering import PaperClusterer
from utils.document_retriever import DocumentRetriever
from utils.hallucination_guard import HallucinationGuard

from agents.claim_extraction import ClaimExtractor
from agents.evidence_collection import EvidenceCollector
from agents.reliability_analysis import ReliabilityAnalyzer
from agents.ranking_prioritization import RankingPrioritizer
from agents.agent4_agreement_disagreement import AgreementDetector
from agents.agent5_uncertainty_gap import UncertaintyDetector
from orchestration.execution_monitor import ExecutionMonitor
from monitoring.logger import get_logger
from monitoring.performance_tracker import PerformanceTracker
from evaluation.evaluator import Evaluator
from export.csv_exporter import export_ranked_insights_csv
from export.pdf_exporter import export_report_pdf

load_dotenv()

_CLAIM_QUERY = "main claims findings results contributions conclusions"


def _safe_metadata_from_paper(paper: dict) -> dict:
    return {
        "citations": paper.get("citations", 0),
        "year": paper.get("year", 0),
        "journal": paper.get("journal", False),
        "dataset_size": paper.get("dataset_size", 0),
    }


def _run_monitored(agent_name, fn, monitor, performance_tracker, logger, *args, **kwargs):
    monitor.start(agent_name)
    logger.info(f"Starting {agent_name}")
    try:
        result = fn(*args, **kwargs)
        monitor.stop(agent_name, status="success")
        duration = monitor.get(agent_name).get("duration_sec", 0.0) or 0.0
        performance_tracker.record(agent_name, duration, True)
        logger.info(f"Completed {agent_name}")
        return result
    except Exception as e:
        monitor.stop(agent_name, status="failed", error=str(e))
        duration = monitor.get(agent_name).get("duration_sec", 0.0) or 0.0
        performance_tracker.record(agent_name, duration, False)
        logger.exception(f"{agent_name} failed: {str(e)}")
        raise


def run_pipeline(
    query: str,
    top_k_per_source: int = 10,
    final_top_k: int = 15,
    debug: bool = True,
) -> dict:
    logger = get_logger("main_pipeline")
    monitor = ExecutionMonitor()
    performance_tracker = PerformanceTracker()
    errors = []

    api_keys = load_groq_api_keys()

    if debug:
        print("\n=== STEP 1: Query Preprocessing ===")
    preprocessor = QueryPreprocessor(api_key=api_keys, debug=debug)
    processed = _run_monitored(
        "step_1_query_preprocessing",
        preprocessor.process,
        monitor,
        performance_tracker,
        logger,
        query,
    )
    cleaned_query = processed["cleaned_query"]
    domains = processed["domains"]

    if debug:
        print(f"Cleaned query : {cleaned_query}")
        print(f"Domains       : {domains}")

    if debug:
        print("\n=== STEP 2: Retrieval ===")
    retriever = Retriever(top_k_per_source=top_k_per_source, debug=debug)
    raw_papers = _run_monitored(
        "step_2_retrieval",
        retriever.retrieve,
        monitor,
        performance_tracker,
        logger,
        query=cleaned_query,
        domains=domains,
    )

    if debug:
        print(f"Raw papers retrieved: {len(raw_papers)}")

    if not raw_papers:
        print("[main] No papers retrieved. Exiting.")
        return {
            "query": query,
            "cleaned_query": cleaned_query,
            "domains": domains,
            "papers": [],
            "results": [],
            "reliability_results": [],
            "agreement_results": [],
            "uncertainty_results": [],
            "ranked_insights": [],
            "execution": monitor.all(),
            "performance_summary": performance_tracker.summary(),
            "errors": errors,
        }

    if debug:
        print("\n=== STEP 3: Embedding + Aggregation ===")
    engine = EmbeddingEngine(debug=debug)
    aggregator = Aggregator(
        embedding_engine=engine,
        final_top_k=final_top_k,
        debug=debug
    )

    raw_papers = _run_monitored(
        "step_3_embedding",
        engine.process,
        monitor,
        performance_tracker,
        logger,
        raw_papers,
        cleaned_query,
        keep_embeddings=True,
    )

    final_papers = _run_monitored(
        "step_3_aggregation",
        aggregator.aggregate,
        monitor,
        performance_tracker,
        logger,
        raw_papers,
        query=cleaned_query,
    )

    if debug:
        print(f"Papers after aggregation: {len(final_papers)}")

    if not final_papers:
        print("[main] No papers passed aggregation threshold. Exiting.")
        return {
            "query": query,
            "cleaned_query": cleaned_query,
            "domains": domains,
            "papers": [],
            "results": [],
            "reliability_results": [],
            "agreement_results": [],
            "uncertainty_results": [],
            "ranked_insights": [],
            "execution": monitor.all(),
            "performance_summary": performance_tracker.summary(),
            "errors": errors,
        }

    if debug:
        print("\n=== STEP 4: Clustering ===")
    clusterer = PaperClusterer(embedding_engine=engine, debug=debug)
    final_papers = _run_monitored(
        "step_4_clustering",
        clusterer.cluster,
        monitor,
        performance_tracker,
        logger,
        final_papers,
    )

    for p in final_papers:
        p.pop("embedding", None)

    if debug:
        print("\n=== STEP 5: Claim Extraction (Agent 1) ===")
    claim_extractor = ClaimExtractor(api_key=api_keys, model=DEFAULT_CLAIM_EXTRACTION_MODEL)
    guard = HallucinationGuard(api_key=api_keys, model=DEFAULT_CLAIM_EXTRACTION_MODEL, debug=debug)
    doc_retriever = DocumentRetriever(model=engine.model, debug=debug)

    monitor.start("agent_1_claim_extraction")
    logger.info("Starting agent_1_claim_extraction")

    try:
        for paper in final_papers:
            pdf_url = paper.get("pdf_url")

            if pdf_url and doc_retriever.load_from_url(pdf_url):
                context = doc_retriever.retrieve(_CLAIM_QUERY, top_k=4)
                if debug:
                    print(f"  [RAG] {paper.get('paper_id')} — full text ({doc_retriever._chunks.__len__()} chunks)")
            else:
                context = paper.get("abstract", "")
                if debug:
                    print(f"  [ABSTRACT] {paper.get('paper_id')}")

            claim_result = claim_extractor.extract(context)
            extracted_claim = claim_result.get("claim")
            verification = guard.verify_claim(extracted_claim or "", context)

            paper["claim_subject"] = claim_result.get("subject")
            paper["claim_predicate"] = claim_result.get("predicate")
            paper["claim_object"] = claim_result.get("object")
            paper["raw_extracted_claim"] = extracted_claim
            paper["claim_grounded"] = verification.get("grounded", False)
            paper["claim_grounding_score"] = verification.get("score", 0.0)
            paper["claim_grounding_method"] = verification.get("method")
            paper["claim_grounding_reasoning"] = verification.get("reasoning")
            paper["claim_grounding_confidence"] = verification.get("confidence")
            paper["extracted_claim"] = extracted_claim if verification.get("grounded", False) else None
            paper["claim_confidence"] = claim_result.get("confidence")
            paper["claim_reasoning"] = claim_result.get("reasoning")

            if debug:
                claim_preview = paper.get("extracted_claim") or "FAILED"
                print(
                    f"    [GUARD] grounded={paper.get('claim_grounded')} "
                    f"method={paper.get('claim_grounding_method')} "
                    f"score={paper.get('claim_grounding_score', 0.0):.2f}"
                )
                print(f"  [{paper.get('paper_id')}] {claim_preview[:80]}")

        monitor.stop("agent_1_claim_extraction", status="success")
        duration = monitor.get("agent_1_claim_extraction").get("duration_sec", 0.0) or 0.0
        performance_tracker.record("agent_1_claim_extraction", duration, True)
        logger.info("Completed agent_1_claim_extraction")

    except Exception as e:
        monitor.stop("agent_1_claim_extraction", status="failed", error=str(e))
        duration = monitor.get("agent_1_claim_extraction").get("duration_sec", 0.0) or 0.0
        performance_tracker.record("agent_1_claim_extraction", duration, False)
        logger.exception(f"agent_1_claim_extraction failed: {str(e)}")
        errors.append({"agent": "agent_1_claim_extraction", "error": str(e)})

    if debug:
        print("\n=== STEP 6: Evidence Collection (Agent 2) ===")

    results = []
    collector = EvidenceCollector(api_key=api_keys, model=DEFAULT_EVIDENCE_COLLECTION_MODEL)

    monitor.start("agent_2_evidence_collection")
    logger.info("Starting agent_2_evidence_collection")

    try:
        for i, focal_paper in enumerate(final_papers):
            claim = focal_paper.get("extracted_claim")
            if not claim:
                continue

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
            paper_lookup = {p.get("paper_id"): p.get("abstract", "") for p in other_papers}

            def _filter_grounded_evidence(items: list[dict]) -> list[dict]:
                filtered = []
                for item in items:
                    abstract = paper_lookup.get(item.get("paper_id"), "")
                    verification = guard.verify_evidence_span_reasoning(
                        reasoning=item.get("reasoning", ""),
                        abstract=abstract,
                        evidence_span=item.get("evidence_span", ""),
                    )
                    if verification.get("grounded", False):
                        item["grounding"] = verification
                        filtered.append(item)
                return filtered

            grounded_supporting = _filter_grounded_evidence(evidence.get("supporting", []))
            grounded_contradicting = _filter_grounded_evidence(evidence.get("contradicting", []))
            grounded_inconclusive = _filter_grounded_evidence(evidence.get("inconclusive", []))

            results.append({
                "focal_paper_id": focal_paper.get("paper_id"),
                "focal_paper_title": focal_paper.get("title"),
                "claim": claim,
                "claim_structured": {
                    "subject": focal_paper.get("claim_subject"),
                    "predicate": focal_paper.get("claim_predicate"),
                    "object": focal_paper.get("claim_object"),
                },
                "claim_confidence": focal_paper.get("claim_confidence"),
                "claim_grounding": {
                    "grounded": focal_paper.get("claim_grounded"),
                    "score": focal_paper.get("claim_grounding_score"),
                    "method": focal_paper.get("claim_grounding_method"),
                    "reasoning": focal_paper.get("claim_grounding_reasoning"),
                    "confidence": focal_paper.get("claim_grounding_confidence"),
                },
                "cluster_id": focal_paper.get("cluster_id"),
                "cluster_label": focal_paper.get("cluster_label"),
                "supporting": grounded_supporting,
                "contradicting": grounded_contradicting,
                "inconclusive": grounded_inconclusive,
            })

            if debug:
                print(
                    f"  [{focal_paper.get('paper_id')}] "
                    f"support={len(grounded_supporting)} "
                    f"contradict={len(grounded_contradicting)} "
                    f"inconclusive={len(grounded_inconclusive)}"
                )

        monitor.stop("agent_2_evidence_collection", status="success")
        duration = monitor.get("agent_2_evidence_collection").get("duration_sec", 0.0) or 0.0
        performance_tracker.record("agent_2_evidence_collection", duration, True)
        logger.info("Completed agent_2_evidence_collection")

    except Exception as e:
        monitor.stop("agent_2_evidence_collection", status="failed", error=str(e))
        duration = monitor.get("agent_2_evidence_collection").get("duration_sec", 0.0) or 0.0
        performance_tracker.record("agent_2_evidence_collection", duration, False)
        logger.exception(f"agent_2_evidence_collection failed: {str(e)}")
        errors.append({"agent": "agent_2_evidence_collection", "error": str(e)})

    if debug:
        print("\n=== STEP 7: Reliability Analysis (Agent 3) ===")

    reliability_results = []
    try:
        reliability_agent = ReliabilityAnalyzer()
        monitor.start("agent_3_reliability_analysis")
        logger.info("Starting agent_3_reliability_analysis")

        for paper in final_papers:
            paper_text = paper.get("abstract", "") or ""
            metadata = _safe_metadata_from_paper(paper)
            reliability_output = reliability_agent.evaluate(paper_text, metadata)
            reliability_results.append({
                "paper_id": paper.get("paper_id"),
                "title": paper.get("title"),
                "reliability": reliability_output,
            })

        monitor.stop("agent_3_reliability_analysis", status="success")
        duration = monitor.get("agent_3_reliability_analysis").get("duration_sec", 0.0) or 0.0
        performance_tracker.record("agent_3_reliability_analysis", duration, True)
        logger.info("Completed agent_3_reliability_analysis")

    except Exception as e:
        monitor.stop("agent_3_reliability_analysis", status="failed", error=str(e))
        duration = monitor.get("agent_3_reliability_analysis").get("duration_sec", 0.0) or 0.0
        performance_tracker.record("agent_3_reliability_analysis", duration, False)
        logger.exception(f"agent_3_reliability_analysis failed: {str(e)}")
        errors.append({"agent": "agent_3_reliability_analysis", "error": str(e)})

    if debug:
        print("\n=== STEP 8: Agreement Detection (Agent 4) ===")

    agreement_results = []
    try:
        agreement_agent = AgreementDetector()
        monitor.start("agent_4_agreement_detection")
        logger.info("Starting agent_4_agreement_detection")

        agreement_results = agreement_agent.detect(results)

        monitor.stop("agent_4_agreement_detection", status="success")
        duration = monitor.get("agent_4_agreement_detection").get("duration_sec", 0.0) or 0.0
        performance_tracker.record("agent_4_agreement_detection", duration, True)
        logger.info("Completed agent_4_agreement_detection")

    except Exception as e:
        monitor.stop("agent_4_agreement_detection", status="failed", error=str(e))
        duration = monitor.get("agent_4_agreement_detection").get("duration_sec", 0.0) or 0.0
        performance_tracker.record("agent_4_agreement_detection", duration, False)
        logger.exception(f"agent_4_agreement_detection failed: {str(e)}")
        errors.append({"agent": "agent_4_agreement_detection", "error": str(e)})

    if debug:
        print("\n=== STEP 9: Uncertainty Detection (Agent 5) ===")

    uncertainty_results = []
    try:
        uncertainty_agent = UncertaintyDetector()
        monitor.start("agent_5_uncertainty_detection")
        logger.info("Starting agent_5_uncertainty_detection")

        uncertainty_results = uncertainty_agent.detect(results)

        monitor.stop("agent_5_uncertainty_detection", status="success")
        duration = monitor.get("agent_5_uncertainty_detection").get("duration_sec", 0.0) or 0.0
        performance_tracker.record("agent_5_uncertainty_detection", duration, True)
        logger.info("Completed agent_5_uncertainty_detection")

    except Exception as e:
        monitor.stop("agent_5_uncertainty_detection", status="failed", error=str(e))
        duration = monitor.get("agent_5_uncertainty_detection").get("duration_sec", 0.0) or 0.0
        performance_tracker.record("agent_5_uncertainty_detection", duration, False)
        logger.exception(f"agent_5_uncertainty_detection failed: {str(e)}")
        errors.append({"agent": "agent_5_uncertainty_detection", "error": str(e)})

    if debug:
        print("\n=== STEP 10: Ranking + Prioritization (Agent 6) ===")

    ranked_insights = []
    try:
        ranking_agent = RankingPrioritizer()
        monitor.start("agent_6_ranking_prioritization")
        logger.info("Starting agent_6_ranking_prioritization")

        ranked_insights = ranking_agent.rank(
            claims=results,
            evidence=results,
            reliability=reliability_results,
            agreements=agreement_results,
            uncertainties=uncertainty_results,
        )

        monitor.stop("agent_6_ranking_prioritization", status="success")
        duration = monitor.get("agent_6_ranking_prioritization").get("duration_sec", 0.0) or 0.0
        performance_tracker.record("agent_6_ranking_prioritization", duration, True)
        logger.info("Completed agent_6_ranking_prioritization")

    except Exception as e:
        monitor.stop("agent_6_ranking_prioritization", status="failed", error=str(e))
        duration = monitor.get("agent_6_ranking_prioritization").get("duration_sec", 0.0) or 0.0
        performance_tracker.record("agent_6_ranking_prioritization", duration, False)
        logger.exception(f"agent_6_ranking_prioritization failed: {str(e)}")
        errors.append({"agent": "agent_6_ranking_prioritization", "error": str(e)})

    output = {
        "query": query,
        "cleaned_query": cleaned_query,
        "domains": domains,
        "papers": final_papers,
        "results": results,
        "reliability_results": reliability_results,
        "agreement_results": agreement_results,
        "uncertainty_results": uncertainty_results,
        "ranked_insights": ranked_insights,
        "execution": monitor.all(),
        "performance_summary": performance_tracker.summary(),
        "errors": errors,
    }

    if debug:
        print(f"\n=== PIPELINE COMPLETE ===")
        print(f"Papers analysed      : {len(final_papers)}")
        print(f"Claims extracted     : {len([p for p in final_papers if p.get('extracted_claim')])}")
        print(f"Evidence results     : {len(results)}")
        print(f"Reliability entries  : {len(reliability_results)}")
        print(f"Agreement results    : {len(agreement_results) if isinstance(agreement_results, list) else 0}")
        print(f"Uncertainty results  : {len(uncertainty_results) if isinstance(uncertainty_results, list) else 0}")
        print(f"Ranked insights      : {len(ranked_insights) if isinstance(ranked_insights, list) else 0}")
        print(f"Errors               : {len(errors)}")

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Agent Scientific Literature Analysis")
    parser.add_argument("query", type=str, help="Research query to analyse")
    parser.add_argument("--top-k", type=int, default=10, help="Papers per source (default: 10)")
    parser.add_argument("--final-k", type=int, default=15, help="Final papers after aggregation (default: 15)")
    parser.add_argument("--output", type=str, default=None, help="Save output to JSON file")
    parser.add_argument("--export-csv", action="store_true", help="Export ranked insights as CSV")
    parser.add_argument("--export-pdf", action="store_true", help="Export final report as PDF")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation framework")
    parser.add_argument("--expected-topics", nargs="*", default=[], help="Expected topics for evaluation")
    parser.add_argument("--quiet", action="store_true", help="Suppress debug output")
    args = parser.parse_args()

    output = run_pipeline(
        query=args.query,
        top_k_per_source=args.top_k,
        final_top_k=args.final_k,
        debug=not args.quiet,
    )

    if args.evaluate:
        evaluator = Evaluator()
        evaluation_report = evaluator.evaluate(output, expected_topics=args.expected_topics)
        output["evaluation_report"] = evaluation_report
        print("\n--- Evaluation Report ---")
        print(json.dumps(evaluation_report, indent=2, default=str))

    if args.export_csv:
        csv_path = export_ranked_insights_csv(output)
        print(f"\nCSV exported to: {csv_path}")

    if args.export_pdf:
        pdf_path = export_report_pdf(output)
        print(f"\nPDF exported to: {pdf_path}")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\nOutput saved to {args.output}")
    else:
        print("\n--- Final JSON Output ---")
        print(json.dumps(output, indent=2, default=str))