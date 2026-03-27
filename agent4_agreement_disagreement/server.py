"""
server.py — Agent 4 Bridge Server
===================================
Serves the HTML frontend and exposes /api/analyze using the
Python backend (agent_4_agreement_disagreement + core).

Run:  python server.py
Then: open http://localhost:5004 in your browser.

The HTML frontend auto-detects this server and routes analysis
through the Python backend instead of calling Mistral directly.
Optionally pass a Mistral API key for AI-enhanced mode.
"""

import os
import sys
import json
import math
import re
from collections import defaultdict
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_4_agreement_disagreement.agent import run_agent_4
from core.orchestrator import load_papers

PORT = 5004
HTML_PATH = os.path.join(os.path.dirname(__file__), "interface", "dashboard.html")


# ── Schema adapter: Python output → HTML expected schema ──────────────────────

def _corpus_synthesis(papers: dict, comparisons: list, summary: dict) -> str:
    paper_names = list(papers.keys())
    agree = summary.get("agreement_count", 0)
    contra = summary.get("contradiction_count", 0)
    total = summary.get("total_comparisons", 1) or 1
    agree_pct = round(agree / total * 100)
    contra_pct = round(contra / total * 100)
    return (
        f"Across {len(paper_names)} papers ({', '.join(paper_names)}), the corpus reveals "
        f"{agree_pct}% agreement and {contra_pct}% contradiction across {total} claim pairs. "
        f"The average confidence score of {summary.get('avg_confidence', 0):.0%} suggests "
        f"{'strong' if summary.get('avg_confidence', 0) > 0.75 else 'moderate'} analytical certainty. "
        f"Key open tensions remain between the methodological and performance claims."
    )

def _highest_impact_problem(comparisons: list) -> str:
    contradictions = [c for c in comparisons if c.get("relationship") == "contradiction"]
    if contradictions:
        top = sorted(contradictions, key=lambda x: -x.get("confidence", 0))[0]
        return (
            f"The most critical unresolved tension concerns: \"{top.get('claim', '')[:120]}\". "
            f"Papers {' and '.join(top.get('papers', []))} reach opposing conclusions; "
            "a controlled benchmark study with shared evaluation protocols is urgently needed."
        )
    novels = [c for c in comparisons if c.get("relationship") == "novel"]
    if novels:
        top = novels[0]
        return (
            f"The corpus's most significant frontier: \"{top.get('claim', '')[:120]}\". "
            "This claim lacks corroboration across the corpus and represents the highest-leverage "
            "target for independent replication studies."
        )
    return (
        "The primary open challenge is establishing a unified benchmark that allows "
        "direct empirical comparison of the methodological approaches described across these papers."
    )

def adapt_agent4_output(result: dict, papers: dict) -> dict:
    """Transform Python backend output to match the HTML frontend schema."""
    comparisons_raw = result.get("comparisons", [])
    adapted_comparisons = []
    for c in comparisons_raw:
        pair = c.get("claim_pair", {})
        adapted_comparisons.append({
            "claim":             c.get("claim", ""),
            "papers":            c.get("papers", []),
            "relationship":      c.get("relationship", "novel"),
            "confidence":        c.get("confidence", 0.5),
            "evidence":          c.get("evidence", []),
            "claim1_category":   pair.get("claim1_category", c.get("category", "general")),
            "claim2_category":   pair.get("claim2_category", c.get("category", "general")),
            "semantic_similarity": c.get("semantic_similarity", 0.0),
            "explanation":       c.get("explanation", ""),
            "research_gap":      c.get("research_gap", ""),
        })

    summary_raw = result.get("summary", {})
    summary = dict(summary_raw)
    summary["corpus_synthesis"]            = _corpus_synthesis(papers, adapted_comparisons, summary_raw)
    summary["highest_impact_open_problem"] = _highest_impact_problem(adapted_comparisons)

    return {
        "claims_by_paper": result.get("claims_by_paper", {}),
        "comparisons":     adapted_comparisons,
        "summary":         summary,
    }


# ── HTTP handler ──────────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print(f"[Server] {self.address_string()} — {fmt % args}")

    def _send_json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self):
        with open(HTML_PATH, "rb") as f:
            body = f.read()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._send_html()
        elif path == "/api/health":
            self._send_json({"status": "ok", "agent": "Agent 4 — Agreement & Disagreement Analyst"})
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        path = urlparse(self.path).path
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)

        if path == "/api/analyze":
            try:
                payload = json.loads(body)
                papers_list = payload.get("papers", [])
                if not papers_list:
                    self._send_json({"error": "No papers provided."}, 400)
                    return
                papers = {p["name"]: p["text"] for p in papers_list}
                print(f"[Server] Running Agent 4 on {len(papers)} paper(s)…")
                raw_result = run_agent_4(papers)
                result = adapt_agent4_output(raw_result, papers)
                self._send_json(result)
            except Exception as e:
                print(f"[Server] Error: {e}")
                self._send_json({"error": str(e)}, 500)
        else:
            self.send_response(404)
            self.end_headers()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not os.path.exists(HTML_PATH):
        print(f"[Error] HTML not found at {HTML_PATH}")
        sys.exit(1)

    print("=" * 60)
    print("  Agent 4 — Agreement & Disagreement Analyst")
    print("  Local Backend Server")
    print("=" * 60)
    print(f"\n  Open in browser → http://localhost:{PORT}")
    print("  Press Ctrl+C to stop.\n")

    server = HTTPServer(("localhost", PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[Server] Stopped.")
