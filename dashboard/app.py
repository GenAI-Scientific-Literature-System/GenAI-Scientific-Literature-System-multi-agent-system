from flask import Flask, request, jsonify

from orchestration.orchestrator import MultiAgentOrchestrator
from evaluation.evaluator import Evaluator
from export.csv_exporter import export_ranked_insights_csv
from export.pdf_exporter import export_report_pdf

from dashboard.components.ranked_insights import build_ranked_insights_view
from dashboard.components.consensus_conflicts import build_consensus_conflicts_view
from dashboard.components.system_status import build_system_status_view
from dashboard.components.performance_panel import build_performance_monitor_view

app = Flask(__name__)

orchestrator = MultiAgentOrchestrator()
evaluator = Evaluator()

LAST_RESULT = {}


@app.route("/")
def home():
    return jsonify({
        "message": "Scientific Literature Multi-Agent Dashboard API is running"
    })


@app.route("/run", methods=["POST"])
def run_pipeline():
    global LAST_RESULT

    data = request.get_json(silent=True) or {}
    query = data.get("query", "")
    papers = data.get("papers", [])

    if not query:
        return jsonify({"message": "Query is required"}), 400

    result = orchestrator.run(query, papers)
    LAST_RESULT = result

    return jsonify(result)


@app.route("/status", methods=["GET"])
def status():
    if not LAST_RESULT:
        return jsonify({"message": "No run available yet"}), 404

    return jsonify(build_system_status_view(LAST_RESULT))


@app.route("/ranked-insights", methods=["GET"])
def ranked_insights():
    if not LAST_RESULT:
        return jsonify({"message": "No run available yet"}), 404

    return jsonify(build_ranked_insights_view(LAST_RESULT))


@app.route("/consensus-conflicts", methods=["GET"])
def consensus_conflicts():
    if not LAST_RESULT:
        return jsonify({"message": "No run available yet"}), 404

    return jsonify(build_consensus_conflicts_view(LAST_RESULT))


@app.route("/performance", methods=["GET"])
def performance():
    if not LAST_RESULT:
        return jsonify({"message": "No run available yet"}), 404

    return jsonify(build_performance_monitor_view(LAST_RESULT))


@app.route("/evaluate", methods=["POST"])
def evaluate_run():
    if not LAST_RESULT:
        return jsonify({"message": "No run available yet"}), 404

    data = request.get_json(silent=True) or {}
    expected_topics = data.get("expected_topics", [])

    report = evaluator.evaluate(LAST_RESULT, expected_topics=expected_topics)
    return jsonify(report)


@app.route("/export/csv", methods=["GET"])
def export_csv():
    if not LAST_RESULT:
        return jsonify({"message": "No run available yet"}), 404

    path = export_ranked_insights_csv(LAST_RESULT)
    return jsonify({"csv_path": path})


@app.route("/export/pdf", methods=["GET"])
def export_pdf():
    if not LAST_RESULT:
        return jsonify({"message": "No run available yet"}), 404

    path = export_report_pdf(LAST_RESULT)
    return jsonify({"pdf_path": path})


@app.route("/dashboard", methods=["GET"])
def dashboard_summary():
    if not LAST_RESULT:
        return jsonify({"message": "No run available yet"}), 404

    return jsonify({
        "system_status": build_system_status_view(LAST_RESULT),
        "ranked_insights": build_ranked_insights_view(LAST_RESULT),
        "consensus_conflicts": build_consensus_conflicts_view(LAST_RESULT),
        "performance": build_performance_monitor_view(LAST_RESULT)
    })


if __name__ == "__main__":
    app.run(debug=True, port=5001)