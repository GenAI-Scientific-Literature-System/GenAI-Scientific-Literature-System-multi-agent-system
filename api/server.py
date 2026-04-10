"""
MERLIN Flask API
Endpoints: health, upload (PDF), analyse, sample, cache/clear
"""
import logging
import os

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

_load_dotenv_early()

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from src.pipeline import run_pipeline
from src.pdf_extractor import extract_text_from_pdf, is_pdf_available
from src.mistral_client import clear_cache
from src.document_store  import clear as clear_doc_store

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


# ── Static ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")

@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(FRONTEND_DIR, filename)


# ── Health ────────────────────────────────────────────────────────────────────
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status":       "ok",
        "version":      "1.0.0",
        "model":        "MERLIN",
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
    """
    POST /api/analyse   application/json
    Body: { "papers": [{"id": "p1", "text": "..."}] }
    """
    data = request.get_json(silent=True)
    if not data or "papers" not in data:
        return jsonify({"error": "Missing 'papers' field."}), 400

    papers = data["papers"]
    if not isinstance(papers, list) or len(papers) == 0:
        return jsonify({"error": "papers must be a non-empty list."}), 400

    papers = papers[:5]
    try:
        result = run_pipeline(papers)
        return jsonify(result.to_dict())
    except Exception as e:
        logger.exception("Pipeline error: %s", e)
        return jsonify({"error": str(e)}), 500


# ── Cache ─────────────────────────────────────────────────────────────────────
@app.route("/api/cache/clear", methods=["POST"])
def clear():
    clear_cache()
    clear_doc_store()
    return jsonify({"status": "cache cleared"})


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
    logger.info("Starting MERLIN API on %s:%d", API_HOST, API_PORT)
    app.run(host=API_HOST, port=API_PORT, debug=DEBUG)
