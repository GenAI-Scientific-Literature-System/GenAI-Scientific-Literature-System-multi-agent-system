"""
run_agent4.py
==============
Entry point: load papers → run Agent 4 → save results → launch dashboard.
"""

import sys
import os
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_4_agreement_disagreement.agent import run_agent_4
from core.orchestrator import load_papers, save_results, save_graph

try:
    from agent_4_agreement_disagreement.claim_graph import build_claim_graph, graph_to_dict, compute_graph_metrics
    GRAPH_AVAILABLE = True
except ImportError:
    GRAPH_AVAILABLE = False


def main():
    print("=" * 60)
    print("  AGENT 4 — Agreement & Disagreement Analyst")
    print("  Multi-Agent Research Intelligence System")
    print("=" * 60)

    # 1. Load papers
    papers = load_papers()
    print(f"\n[Main] Loaded {len(papers)} papers: {list(papers.keys())}\n")

    # 2. Run Agent 4
    results = run_agent_4(papers)

    # 3. Save results
    out_path = save_results(results)
    print(f"\n[Main] Results saved to: {out_path}")

    # 4. Build & save claim graph
    if GRAPH_AVAILABLE:
        try:
            G = build_claim_graph(results["comparisons"])
            gdict = graph_to_dict(G)
            metrics = compute_graph_metrics(G)
            save_graph(gdict)
            print(f"\n[Main] Graph metrics: {json.dumps(metrics, indent=2)}")
        except Exception as e:
            print(f"[Main] Graph build skipped: {e}")

    # 5. Print summary
    s = results["summary"]
    print("\n" + "=" * 60)
    print("  ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"  Total Comparisons  : {s['total_comparisons']}")
    print(f"  Agreements         : {s['agreement_count']} ({s['agreement_ratio']:.1%})")
    print(f"  Contradictions     : {s['contradiction_count']} ({s['contradiction_ratio']:.1%})")
    print(f"  Partial Agreements : {s['partial_count']}")
    print(f"  Novel Claims       : {s['novel_count']}")
    print(f"  Avg Confidence     : {s['avg_confidence']:.1%}")
    print("=" * 60)
    print("\n✅ Analysis complete.")
    print("   Run the dashboard: streamlit run interface/dashboard.py")
    print("   Or use run.bat to launch automatically.\n")


if __name__ == "__main__":
    main()
