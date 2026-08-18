"""
GLAS-Med Flask API
Endpoints: health, upload (PDF), analyse, sample, cache/clear
"""
import logging
import os
from collections import defaultdict, deque
import math
import re

# ── Load .env before any other imports touch os.environ ──────────────────────
def _load_dotenv_early():
    try:
        from dotenv import load_dotenv
        _here = os.path.dirname(os.path.abspath(__file__))
        for _candidate in [
            os.path.join(_here, "..", "..", ".env"),   # integrated/.env
            os.path.join(_here, "..", ".env"),
            os.path.join(os.getcwd(), ".env"),
        ]:
            if os.path.isfile(_candidate):
                load_dotenv(_candidate, override=False)
                break
    except ImportError:
        pass

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from src.pipeline import run_pipeline
from src.pdf_extractor import extract_text_from_pdf, is_pdf_available
from src.llm_client import clear_cache
from src.document_store  import clear as clear_doc_store
from pipeline.retrieval import Retriever
from pipeline.embedding import EmbeddingEngine
from pipeline.aggregator import Aggregator
from export.csv_exporter import export_ranked_insights_csv
from export.pdf_exporter import export_report_pdf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

app = Flask(__name__, static_folder=FRONTEND_DIR)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB per upload
CORS(app)

LAST_ANALYSIS_RESULT = {}

SEARCH_REVIEW_PER_SOURCE_CAP = 5
SEARCH_REVIEW_TOTAL_CAP = 10
SEARCH_REVIEW_MIN_RETRIEVAL = 0.5
SEARCH_REVIEW_RETRIEVAL_DELTA = 20.0
SEARCH_REVIEW_MIN_COVERAGE = 0.08
SEARCH_REVIEW_REL_COVERAGE = 0.25
SEARCH_REVIEW_MIN_TITLE_COVERAGE = 0.03
SEARCH_REVIEW_REL_TITLE_COVERAGE = 0.30
SEARCH_REVIEW_SCORE_RATIO = 0.30
SEARCH_REVIEW_MIN_SCORE = 0.8


def _normalize_retrieved_paper(paper: dict) -> dict | None:
    text = (paper.get("abstract") or "").strip()
    if not text:
        return None

    raw_id = (
        paper.get("paper_id")
        or paper.get("doi")
        or paper.get("title")
        or f"paper_{abs(hash(text)) % 10_000_000}"
    )
    return {
        "id": str(raw_id)[:120],
        "paper_id": str(raw_id)[:120],
        "abstract": text,
        "text": text,
        "title": paper.get("title", ""),
        "source": paper.get("source", ""),
        "year": paper.get("year"),
        "doi": paper.get("doi", ""),
        "url": paper.get("url") or (f"https://doi.org/{paper.get('doi')}" if paper.get("doi") else ""),
        "pdf_url": paper.get("pdf_url", ""),
        "citation_count": paper.get("citation_count") or 0,
        "retrieval_score": paper.get("retrieval_score"),
        "query_coverage": paper.get("query_coverage"),
        "title_query_coverage": paper.get("title_query_coverage"),
        "phrase_hits": paper.get("phrase_hits"),
        "title_phrase_hits": paper.get("title_phrase_hits"),
        "journal": paper.get("journal"),
        "authors": paper.get("authors") or [],
        "categories": paper.get("categories") or [],
    }


def _source_review_score(paper: dict) -> float:
    retrieval = float(paper.get("retrieval_score") or 0.0)
    coverage = float(paper.get("query_coverage") or 0.0)
    title_coverage = float(paper.get("title_query_coverage") or 0.0)
    phrase_hits = int(paper.get("phrase_hits") or 0)

    year_boost = 0.0
    year = paper.get("year")
    if isinstance(year, int):
        year_boost = max(0.0, min(0.5, (year - 2018) * 0.04))

    return retrieval + (coverage * 2.8) + (title_coverage * 1.8) + min(0.9, phrase_hits * 0.25) + year_boost


def _select_search_candidates(
    papers: list[dict],
    per_source_cap: int = SEARCH_REVIEW_PER_SOURCE_CAP,
    total_cap: int = SEARCH_REVIEW_TOTAL_CAP,
) -> list[dict]:
    if not papers:
        return []

    for paper in papers:
        paper["score"] = round(_source_review_score(paper), 4)

    best_score = max(float(p.get("score") or 0.0) for p in papers)
    best_retrieval = max(float(p.get("retrieval_score") or 0.0) for p in papers)
    best_coverage = max(float(p.get("query_coverage") or 0.0) for p in papers)
    best_title_coverage = max(float(p.get("title_query_coverage") or 0.0) for p in papers)
    retrieval_floor = max(SEARCH_REVIEW_MIN_RETRIEVAL, best_retrieval - SEARCH_REVIEW_RETRIEVAL_DELTA)
    score_floor = max(SEARCH_REVIEW_MIN_SCORE, best_score * SEARCH_REVIEW_SCORE_RATIO)
    coverage_floor = max(SEARCH_REVIEW_MIN_COVERAGE, best_coverage * SEARCH_REVIEW_REL_COVERAGE)
    title_coverage_floor = max(SEARCH_REVIEW_MIN_TITLE_COVERAGE, best_title_coverage * SEARCH_REVIEW_REL_TITLE_COVERAGE)

    viable = [
        p for p in papers
        if float(p.get("retrieval_score") or 0.0) >= retrieval_floor
        and float(p.get("score") or 0.0) >= score_floor
        and (
            float(p.get("query_coverage") or 0.0) >= coverage_floor
            or float(p.get("title_query_coverage") or 0.0) >= title_coverage_floor
            or int(p.get("phrase_hits") or 0) >= 2
        )
    ]
    if not viable:
        viable = sorted(papers, key=lambda p: float(p.get("score") or 0.0), reverse=True)[:max(3, min(total_cap, len(papers)))]

    by_source: dict[str, list[dict]] = defaultdict(list)
    for paper in viable:
        by_source[str(paper.get("source") or "unknown")].append(paper)

    for source_rows in by_source.values():
        source_rows.sort(key=lambda p: float(p.get("score") or 0.0), reverse=True)

    selected: list[dict] = []
    seen_ids: set[str] = set()

    rounds = max((len(rows) for rows in by_source.values()), default=0)
    for idx in range(rounds):
        for source, source_rows in sorted(by_source.items()):
            if idx >= len(source_rows) or idx >= per_source_cap or len(selected) >= total_cap:
                continue
            paper = source_rows[idx]
            pid = str(paper.get("paper_id") or paper.get("id") or "")
            if pid and pid not in seen_ids:
                selected.append(paper)
                seen_ids.add(pid)
        if len(selected) >= total_cap:
            break

    if len(selected) < total_cap:
        leftovers = sorted(
            [p for p in viable if str(p.get("paper_id") or p.get("id") or "") not in seen_ids],
            key=lambda p: float(p.get("score") or 0.0),
            reverse=True,
        )
        for paper in leftovers:
            if len(selected) >= total_cap:
                break
            selected.append(paper)

    return selected


def _build_export_payload(result: dict) -> dict:
    payload = {
        "query": result.get("query", ""),
        "execution": result.get("execution", {}),
        "errors": result.get("errors", []),
        "ranked_insights": [],
    }

    ranked = result.get("ranked_insights") or []
    if ranked:
        payload["ranked_insights"] = ranked
        return payload

    # Fallback ranking for src.pipeline output: prioritize high-score gaps, then high-uncertainty claims.
    gaps = result.get("gaps") or []
    claims = result.get("claims") or []

    for g in sorted(gaps, key=lambda x: x.get("gap_signals", {}).get("gap_score", 0), reverse=True)[:10]:
        payload["ranked_insights"].append({
            "insight": g.get("gap", ""),
            "score": g.get("gap_signals", {}).get("gap_score", 0),
            "source_paper": ", ".join(g.get("related_claims", [])[:3]),
            "type": "research_gap",
        })

    for c in sorted(claims, key=lambda x: x.get("uncertainty", 0), reverse=True)[:10]:
        payload["ranked_insights"].append({
            "insight": c.get("text") or f"{c.get('subject', '')} {c.get('predicate', '')} {c.get('object', '')}",
            "score": c.get("uncertainty", 0),
            "source_paper": c.get("paper_id", ""),
            "type": "claim",
        })

    return payload


def _infer_domains_from_query(query: str) -> list[str]:
    q = (query or "").lower()
    domains: list[str] = []

    medical_patterns = [
        r"\bpatient(s)?\b", r"\bclinical\b", r"\btherapy\b", r"\bdisease\b",
        r"\bcancer\b", r"\bdrug(s)?\b", r"\bdiagnos(is|tic)\b", r"\btrial(s)?\b",
    ]
    biology_patterns = [
        r"\bgene(s)?\b", r"\bprotein(s)?\b", r"\bcell(s)?\b", r"\bgenom(ic|ics)\b",
        r"\bmicrobiom(e|al)\b", r"\bmetabol(ism|ic)\b", r"\brna\b", r"\bdna\b",
    ]
    ml_patterns = [
        r"\bllm(s)?\b", r"\btransformer(s)?\b", r"\bmachine learning\b", r"\bdeep learning\b",
        r"\bneural\b", r"\bartificial intelligence\b", r"\bai\b", r"\bnlp\b",
    ]

    if any(re.search(p, q) for p in medical_patterns):
        domains.append("medical")
    if any(re.search(p, q) for p in biology_patterns):
        domains.append("biology")
    if any(re.search(p, q) for p in ml_patterns):
        domains.append("ml")

    if not domains:
        return ["medical", "ml"]
    return domains


def _resolve_domains(requested_domains, query: str) -> list[str]:
    allowed = {"medical", "biology", "ml", "general"}
    if isinstance(requested_domains, list) and requested_domains:
        normalized = [str(d).strip().lower() for d in requested_domains if str(d).strip().lower() in allowed]
        if normalized:
            return normalized
    return _infer_domains_from_query(query)


# ── Static ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")

@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(FRONTEND_DIR, filename)


# ── Health ────────────────────────────────────────────────────────────────────

@app.route("/api/config", methods=["GET"])
def get_sys_config():
    from config import GROQ_MODEL, GROQ_API_KEYS
    groq_active = bool(len(GROQ_API_KEYS) > 0)
    return jsonify({
        "model": GROQ_MODEL,
        "groq_active": groq_active,
        "engine_mode": "Live Groq LLaMA-70B" if groq_active else "Grounded Clinical Engine",
    })

@app.route("/api/health", methods=["GET"])
def health():
    from config import GROQ_API_KEYS
    groq_active = bool(len(GROQ_API_KEYS) > 0)
    return jsonify({
        "status":       "ok",
        "version":      "1.0.0",
        "model":        "GLAS-Med",
        "groq_active":  groq_active,
        "engine_mode":  "Live Groq LLaMA-70B" if groq_active else "Grounded Clinical Engine",
        "pdf_support":  is_pdf_available(),
    })


# ── PDF Upload ────────────────────────────────────────────────────────────────
@app.route("/api/upload", methods=["POST"])
def upload():
    """
    POST /api/upload   multipart/form-data
    Field: files[]  — one or more PDF files (max 5, max 32 MB each)
    Returns: { "papers": [{"id": "filename.pdf", "text": "...", "pages": N, "truncated": bool}] }
    """
    if "files[]" not in request.files:
        return jsonify({"error": "No files[] field in request."}), 400

    uploaded = request.files.getlist("files[]")
    if not uploaded:
        return jsonify({"error": "No files received."}), 400

    uploaded = uploaded[:5]   # max 5 PDFs at once
    papers   = []
    errors   = []

    for f in uploaded:
        fname = f.filename or "upload.pdf"

        if not fname.lower().endswith(".pdf"):
            errors.append(f"{fname}: only PDF files are supported.")
            continue

        file_bytes = f.read()
        if len(file_bytes) == 0:
            errors.append(f"{fname}: file is empty.")
            continue

        result = extract_text_from_pdf(file_bytes, fname)

        if result["error"]:
            errors.append(f"{fname}: {result['error']}")
            continue

        if not result["text"].strip():
            errors.append(f"{fname}: no text could be extracted (scanned image PDF?).")
            continue

        papers.append({
            "id":        fname,
            "text":      result["text"],
            "pages":     result["pages"],
            "truncated": result["truncated"],
        })

    if not papers and errors:
        return jsonify({"error": " | ".join(errors)}), 422

    return jsonify({"papers": papers, "warnings": errors})


# ── Analyse ───────────────────────────────────────────────────────────────────
@app.route("/api/analyse", methods=["POST"])
def analyse():
    data = request.get_json(silent=True)
    if not data or "papers" not in data:
        return jsonify({"error": "Missing 'papers' field."}), 400

    papers = data["papers"]
    if not isinstance(papers, list) or len(papers) == 0:
        return jsonify({"error": "papers must be a non-empty list."}), 400

    # Keep runtime bounded, but allow a wider evidence set for cross-paper
    # contradiction detection.
    papers = papers[:10]
    try:
        from src.pipeline import run_pipeline
        result = run_pipeline(papers).to_dict()
        global LAST_ANALYSIS_RESULT
        LAST_ANALYSIS_RESULT = result
        return jsonify(result)
    except Exception as e:
        logger.exception("Pipeline error: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route("/api/logs/stream")
def stream_logs():
    def generate():
        import time, os
        log_path = 'logs/system.log'
        if not os.path.exists(log_path):
            yield "data: No log file found\n\n"
            return
            
        with open(log_path, 'r') as f:
            f.seek(0, 2) # Move to end of file
            # Limit the generator loop so it doesn't hang forever once client disconnects
            checks = 0
            while True:
                line = f.readline()
                if line:
                    yield f"data: {line}\n\n"
                    checks = 0
                else:
                    time.sleep(0.1)
                    checks += 1
                    if checks > 1200: # 2 minutes idle max
                        break
    from flask import Response
    return Response(generate(), mimetype='text/event-stream')
@app.route("/api/search", methods=["POST"])
def search_sources():
    data = request.get_json(silent=True) or {}
    query = (data.get("query") or "").strip()
    if not query:
        return jsonify({"error": "Missing or empty 'query'."}), 400

    try:
        requested_top_k = int(data.get("top_k_per_source", 5))
    except (TypeError, ValueError):
        requested_top_k = 5
    top_k_per_source = max(1, min(20, requested_top_k))

    domains = _resolve_domains(data.get("domains"), query)

    try:
        retriever = Retriever(top_k_per_source=top_k_per_source, debug=False)
        raw_papers = retriever.retrieve(query=query, domains=domains)
        normalized = [_normalize_retrieved_paper(p) for p in raw_papers]
        papers = [p for p in normalized if p]

        if not papers:
            return jsonify({"error": "No papers retrieved for query.", "query": query}), 404

        selected = _select_search_candidates(
            papers,
            per_source_cap=min(SEARCH_REVIEW_PER_SOURCE_CAP, top_k_per_source),
            total_cap=min(SEARCH_REVIEW_TOTAL_CAP, max(4, top_k_per_source + 1)),
        )

        # Recursively remove numpy types
        def _convert_numpy(o):
            import numpy as np
            if isinstance(o, np.ndarray):
                return o.tolist()
            elif isinstance(o, np.floating):
                return float(o)
            elif isinstance(o, np.integer):
                return int(o)
            elif isinstance(o, dict):
                return {k: _convert_numpy(v) for k, v in o.items()}
            elif isinstance(o, list):
                return [_convert_numpy(v) for v in o]
            return o
            
        selected = _convert_numpy(selected)
        
        # Remove embeddings that were converted to lists
        for p in selected:
            p.pop("embedding", None)

        return jsonify({
            "query": query,
            "papers": selected,
            "source_counts": {src: len([p for p in papers if str(p.get("source") or "unknown") == src]) for src in {str(p.get("source") or "unknown") for p in papers}},
        })
    except Exception as e:
        logger.exception("Search error: %s", e)
        return jsonify({"error": str(e)}), 500

@app.route("/api/run-query", methods=["POST"])
def run_query():
    """
    POST /api/run-query   application/json
    Body: {"query": "...", "top_k_per_source": 5, "domains": ["ml", "medical"]}
    Fetches papers from connectors and runs the same analysis pipeline.
    """
    data = request.get_json(silent=True) or {}
    query = (data.get("query") or "").strip()
    if not query:
        return jsonify({"error": "Missing or empty 'query'."}), 400

    try:
        requested_top_k = int(data.get("top_k_per_source", 5))
    except (TypeError, ValueError):
        requested_top_k = 5
    top_k_per_source = max(1, min(20, requested_top_k))

    domains = _resolve_domains(data.get("domains"), query)

    selection_mode = (data.get("selection_mode") or "global").strip().lower()
    if selection_mode not in {"global", "balanced"}:
        selection_mode = "global"

    try:
        retriever = Retriever(top_k_per_source=top_k_per_source, debug=False)
        raw_papers = retriever.retrieve(query=query, domains=domains)
        normalized = [_normalize_retrieved_paper(p) for p in raw_papers]
        papers = [p for p in normalized if p]

        if not papers:
            return jsonify({"error": "No papers retrieved for query.", "query": query}), 404

        # Keep pipeline bounded for responsiveness.
        # Default mode is global top-K ranking irrespective of source.
        engine = EmbeddingEngine(debug=False)
        aggregator = Aggregator(embedding_engine=engine, final_top_k=5, debug=False)
        
        # We don't need 'balanced' vs 'global' anymore, aggregator acts as the semantic global ranker.
        selected = aggregator.aggregate(papers, query)

        # Preserve evidence provenance required for GLAS-Med reliability and
        # uncertainty scoring (for example citation count and journal data).
        papers_for_pipeline = [
            {key: paper.get(key) for key in (
                "id", "text", "pmid", "trial_id", "citation_count", "year",
                "journal", "sample_size", "dataset_size", "study_design",
                "journal_impact_quartile",
            )}
            for paper in selected
        ]
        result = run_pipeline(papers_for_pipeline).to_dict()
        result["query"] = query

        source_map = {
            p["id"]: {
                "source": p.get("source", ""),
                "title": p.get("title", ""),
                "url": p.get("url") or p.get("pdf_url") or "",
                "year": p.get("year"),
            }
            for p in selected
        }
        for c in result.get("claims", []):
            info = source_map.get(c.get("paper_id"), {})
            c["paper_source"] = info.get("source", "")
            c["paper_title"] = info.get("title", "")
            c["paper_url"] = info.get("url", "")

        result["query_context"] = {
            "domains": domains,
            "top_k_per_source": top_k_per_source,
            "selection_mode": selection_mode,
            "retrieved_count": len(raw_papers),
            "used_in_pipeline": len(papers_for_pipeline),
            "papers": papers[:20],
            "pipeline_papers": selected,
        }

        global LAST_ANALYSIS_RESULT
        LAST_ANALYSIS_RESULT = result
        return jsonify(result)
    except Exception as e:
        logger.exception("Run-query error: %s", e)
        return jsonify({"error": str(e)}), 500


# ── Cache ─────────────────────────────────────────────────────────────────────
@app.route("/api/cache/clear", methods=["POST"])
def clear():
    clear_cache()
    clear_doc_store()
    try:
        from pipeline.retrieval import CACHE_DB
        import sqlite3
        with sqlite3.connect(CACHE_DB) as conn:
            conn.execute("DELETE FROM query_cache")
    except Exception:
        pass
    return jsonify({"status": "cache cleared"})


@app.route("/api/export/csv", methods=["POST"])
def export_csv():
    if not LAST_ANALYSIS_RESULT:
        return jsonify({"error": "No analysis result available yet."}), 400
    try:
        filename = export_ranked_insights_csv(_build_export_payload(LAST_ANALYSIS_RESULT))
        return jsonify({"status": "ok", "file": filename})
    except Exception as e:
        logger.exception("CSV export error: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route("/api/export/pdf", methods=["POST"])
def export_pdf():
    if not LAST_ANALYSIS_RESULT:
        return jsonify({"error": "No analysis result available yet."}), 400
    try:
        filename = export_report_pdf(_build_export_payload(LAST_ANALYSIS_RESULT))
        return jsonify({"status": "ok", "file": filename})
    except Exception as e:
        logger.exception("PDF export error: %s", e)
        return jsonify({"error": str(e)}), 500


# ── Sample ────────────────────────────────────────────────────────────────────
@app.route("/api/sample", methods=["GET"])
def sample():
    return jsonify({"papers": [
        {
            "id": "paper_1",
            "text": (
                "We demonstrate that large language models significantly outperform "
                "traditional retrieval methods on question answering tasks when fine-tuned "
                "on domain-specific corpora. Our experiments assume a high-resource setting "
                "with GPU access. Results show 12% F1 improvement over BM25 baseline using "
                "a transformer encoder with attention mechanism. The study is limited to "
                "English text only and assumes clean, well-formatted input data."
            )
        },
        {
            "id": "paper_2",
            "text": (
                "We show that retrieval-augmented generation fails to consistently outperform "
                "BM25 on low-resource domain-specific tasks. Under our experimental conditions "
                "of limited training data and noisy text inputs, traditional IR methods prove "
                "more robust. We assume a low-resource setting with CPU-only infrastructure. "
                "The performance gap narrows significantly when training data is scarce, "
                "suggesting the importance of assumption-aware evaluation."
            )
        },
    ]})


if __name__ == "__main__":
    from config import API_HOST, API_PORT, DEBUG
    logger.info("Starting GLAS-Med API on %s:%d", API_HOST, API_PORT)
    app.run(host=API_HOST, port=API_PORT, debug=DEBUG)
