"""
server.py — Agent 5 Bridge Server
===================================
Serves the HTML frontend and exposes /api/analyze using the
Python backend (agent_5_uncertainty_gap + gap_ranker + core).

Run:  python server.py
Then: open http://localhost:5005 in your browser.

The HTML frontend auto-detects this server and routes analysis
through the Python backend instead of calling Mistral directly.
"""

import os
import sys
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_5_uncertainty_gap.agent import run_agent_5
from agent_5_uncertainty_gap.gap_ranker import rank_gaps, cluster_gaps_by_theme, generate_gap_summary
from core.orchestrator import load_papers

PORT = 5005
HTML_PATH = os.path.join(os.path.dirname(__file__), "interface", "dashboard.html")


# ── Schema adapter: Python output → HTML expected schema ──────────────────────

def _overconfidence_risk(avg_uncertainty: float, strong_claim_count: int = 0) -> str:
    if avg_uncertainty < 0.15:
        return "high"
    elif avg_uncertainty < 0.30:
        return "moderate"
    return "low"

def _field_maturity(paper_analyses: dict, gaps: list, conflicts: list) -> str:
    n = len(paper_analyses)
    avg_u = sum(a.get("avg_uncertainty", 0) for a in paper_analyses.values()) / max(n, 1)
    high_gaps = sum(1 for g in gaps if g.get("priority") in ("HIGH", "CRITICAL"))
    return (
        f"The {n}-paper corpus exhibits an average uncertainty level of {avg_u:.0%}, "
        f"indicating {'early-stage' if avg_u > 0.35 else 'developing'} epistemic maturity. "
        f"{high_gaps} high-priority research gaps signal significant open territory. "
        f"{'Cross-paper uncertainty conflicts (' + str(len(conflicts)) + ') suggest contested methodological ground.' if conflicts else 'Consistent uncertainty levels across papers suggest methodological coherence.'} "
        f"The field shows promise but requires further empirical grounding before consolidation."
    )

def _most_critical_gap(gaps: list) -> str:
    if not gaps:
        return "No critical gaps detected in this corpus."
    top = gaps[0]
    return (
        f"Grand Challenge: {top.get('category', 'Research Gap')} — "
        f"\"{top.get('description', '')[:150]}\". "
        f"Composite priority score: {top.get('composite_score', 0):.2f}. "
        f"Recommendation: {top.get('recommendation', '')}"
    )

def _framing(gap: dict) -> str:
    gt = gap.get("gap_type", "")
    cat = gap.get("category", "Research Gap")
    src = gap.get("paper_source", "")
    framing_map = {
        "future_direction":   f"Call for Papers: Extending {src}'s work — {cat} as a new research track",
        "open_question":      f"Workshop Proposal: Addressing the open question identified in {src}",
        "data_gap":           f"Shared Task: Creating benchmarks to close the data gap from {src}",
        "unexplored":         f"Emerging Area Workshop: Exploring the territory left open by {src}",
        "investigation_needed": f"Research Challenge: Systematic investigation of {src}'s unresolved question",
        "promising_direction":  f"Frontier Track: Pursuing the promising direction outlined in {src}",
        "underexplored":      f"Spotlight: Bringing focus to the understudied area in {src}",
    }
    return framing_map.get(gt, f"Research Agenda Item: Addressing {cat} identified in {src}")

def adapt_agent5_output(result: dict) -> dict:
    """Transform Python backend output to match the HTML frontend schema."""
    paper_analyses_raw = result.get("paper_analyses", {})
    gaps_raw = result.get("research_gaps", [])
    conflicts_raw = result.get("cross_paper_conflicts", [])
    summary_raw = result.get("summary", {})

    # Transform paper_analyses
    paper_analyses = {}
    for pid, a in paper_analyses_raw.items():
        # Rename uncertain_sentences → top_uncertain_sentences, fix inner key names
        top_sents = []
        for s in a.get("uncertain_sentences", [])[:5]:
            top_sents.append({
                "sentence":         s.get("sentence", ""),
                "uncertainty_score": s.get("uncertainty_score", 0),
                "types":            s.get("uncertainty_types", []),
                "aaai_note":        f"Uncertainty signals detected: {', '.join(s.get('uncertainty_types', [])[:3])}. "
                                    f"Reviewer flag: this sentence warrants hedging qualification or empirical support."
            })

        # Build methodology_concerns from signals
        methodology_concerns = []
        for g in a.get("gap_signals", [])[:4]:
            methodology_concerns.append(
                f"Gap signal [{g['type']}]: \"{g['sentence'][:100]}\""
            )

        paper_analyses[pid] = {
            "avg_uncertainty":         a.get("avg_uncertainty", 0),
            "uncertainty_ratio":       a.get("uncertainty_ratio", 0),
            "overconfidence_risk":     _overconfidence_risk(a.get("avg_uncertainty", 0)),
            "dominant_uncertainty_types": a.get("dominant_uncertainty_types", {}),
            "top_uncertain_sentences": top_sents,
            "gap_signals":             a.get("gap_signals", []),
            "methodology_concerns":    methodology_concerns,
            "overconfident_claims":    [],
        }

    # Transform research_gaps — add tier + aaai_framing if missing
    gaps = []
    for g in gaps_raw:
        gg = dict(g)
        if "tier" not in gg:
            score = gg.get("composite_score", 0)
            gg["tier"] = ("🔴 Critical" if score > 0.80 else
                          "🟠 High"     if score > 0.70 else
                          "🟡 Medium"   if score > 0.55 else
                          "🟢 Low")
        if "aaai_framing" not in gg:
            gg["aaai_framing"] = _framing(gg)
        gaps.append(gg)

    # Transform cross_paper_conflicts
    conflicts = []
    for c in conflicts_raw:
        cc = dict(c)
        if "topic_overlap" not in cc:
            cc["topic_overlap"] = "Overlapping research domain — uncertainty level discrepancy detected"
        if "suggested_resolution" not in cc:
            cc["suggested_resolution"] = (
                f"Design a comparative study where both {c['paper_1']} and {c['paper_2']} "
                "are evaluated on the same benchmark with standardised confidence reporting."
            )
        conflicts.append(cc)

    # Build summary
    critical_count = sum(1 for g in gaps if "Critical" in g.get("tier", ""))
    high_count = sum(1 for g in gaps if g.get("priority") in ("HIGH",) and "Critical" not in g.get("tier", ""))
    summary = dict(summary_raw)
    summary["critical_gaps"]             = critical_count
    summary["high_priority_gaps"]        = high_count
    summary["field_maturity_assessment"] = _field_maturity(paper_analyses_raw, gaps, conflicts)
    summary["most_critical_gap"]         = _most_critical_gap(gaps)

    return {
        "paper_analyses":        paper_analyses,
        "research_gaps":         gaps,
        "cross_paper_conflicts": conflicts,
        "summary":               summary,
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
            self._send_json({"status": "ok", "agent": "Agent 5 — Uncertainty & Research Gap Analyst"})
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
                print(f"[Server] Running Agent 5 on {len(papers)} paper(s)…")
                raw_result = run_agent_5(papers)
                # Apply advanced ranking
                raw_result["research_gaps"] = rank_gaps(raw_result["research_gaps"])
                result = adapt_agent5_output(raw_result)
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
    print("  Agent 5 — Uncertainty & Research Gap Analyst")
    print("  Local Backend Server")
    print("=" * 60)
    print(f"\n  Open in browser → http://localhost:{PORT}")
    print("  Press Ctrl+C to stop.\n")

    server = HTTPServer(("localhost", PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[Server] Stopped.")
