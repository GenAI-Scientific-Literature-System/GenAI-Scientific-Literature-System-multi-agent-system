"""
run_agent5.py
==============
Entry point: load papers → run Agent 5 → rank gaps → save results → launch dashboard.
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_5_uncertainty_gap.agent import run_agent_5
from agent_5_uncertainty_gap.gap_ranker import rank_gaps, cluster_gaps_by_theme, generate_gap_summary
from core.orchestrator import load_papers, save_results


def main():
    print("=" * 60)
    print("  AGENT 5 — Uncertainty & Research Gap Analyst")
    print("  Multi-Agent Research Intelligence System")
    print("=" * 60)

    # 1. Load papers
    papers = load_papers()
    print(f"\n[Main] Loaded {len(papers)} papers: {list(papers.keys())}\n")

    # 2. Run Agent 5 core pipeline
    results = run_agent_5(papers)

    # 3. Re-rank gaps with advanced scoring
    print("[Main] Applying advanced gap ranking...")
    results["research_gaps"] = rank_gaps(results["research_gaps"])

    # 4. Cluster gaps by theme
    print("[Main] Clustering gaps by research theme...")
    clusters = cluster_gaps_by_theme(results["research_gaps"])
    cluster_summary = {theme: len(gaps) for theme, gaps in clusters.items()}
    results["gap_clusters"] = cluster_summary

    # 5. Generate summary
    gap_summary = generate_gap_summary(results["research_gaps"], results["paper_analyses"])
    results["gap_summary"] = gap_summary

    # 6. Save results
    out_path = save_results(results)
    print(f"\n[Main] Results saved to: {out_path}")

    # 7. Print summary
    s = results["summary"]
    print("\n" + "=" * 60)
    print("  ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"  Total Papers            : {s['total_papers']}")
    print(f"  Total Gaps              : {s['total_gaps']}")
    print(f"  High Priority Gaps      : {s['high_priority_gaps']}")
    print(f"  Medium Priority Gaps    : {s['medium_priority_gaps']}")
    print(f"  Low Priority Gaps       : {s['low_priority_gaps']}")
    print(f"  Avg Uncertainty Overall : {s['avg_uncertainty_overall']:.1%}")
    print(f"  Cross-Paper Conflicts   : {s['cross_paper_conflicts']}")
    print(f"\n  Theme Clusters: {cluster_summary}")
    print("=" * 60)

    if results["research_gaps"]:
        top = results["research_gaps"][0]
        print(f"\n  🏆 Top Gap: [{top.get('tier','?')}] {top.get('category','')}")
        print(f"     {top.get('description','')[:100]}...")
        print(f"     Composite Score: {top.get('composite_score',0):.3f}")

    if gap_summary.get("key_findings"):
        print("\n  📌 Key Findings:")
        for f in gap_summary["key_findings"]:
            print(f"     • {f}")

    print("\n✅ Analysis complete.")
    print("   Run the dashboard: streamlit run interface/dashboard.py")
    print("   Or use run.bat to launch automatically.\n")


if __name__ == "__main__":
    main()
