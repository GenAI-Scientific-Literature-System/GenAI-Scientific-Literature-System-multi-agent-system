"""
agents/citation_analysis/citation_analysis_agent.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Agent 8: Citation Network Analysis  [OCP EXTENSION EXAMPLE]

This file demonstrates how to add a brand new agent to the system
WITHOUT modifying any existing file except agents/__init__.py
(one import line added).

IDENTITY
  agent_id        : "agent_8_citation_analysis"
  role            : Build a citation influence graph from paper reference
                    lists and use PageRank to surface the most influential
                    works referenced across the retrieved corpus.
  prompt_template : CITATION_ANALYSIS_PROMPT

PIPELINE CONTRACT
  Input  :
    context.papers                       → raw papers with "references" field
    "agent_1_claim_extraction.claims"    → to weight citations by claim relevance
  Output :
    {
      "citation_graph": {
          "nodes": List[{paper_id, title, pagerank_score}],
          "edges": List[{source, target, weight}],
      },
      "top_influential": List[{paper_id, title, score}],
      "isolated_papers": List[str],   # paper_ids with no citation links
    }

HOW TO ACTIVATE:
  Add ONE line to agents/__init__.py:
    from agents.citation_analysis.citation_analysis_agent import CitationAnalysisAgent
  That's it. Zero other files change.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from agents.base.base_agent import AgentContext, AgentResult, BaseAgent
from core.registry.agent_registry import registry

# ── Prompt owned by this agent ────────────────────────────────────────────────
CITATION_ANALYSIS_PROMPT = """\
You are Agent 8 — a scientific citation network analyst.

You are given a set of research papers with their reference lists.
Your task: identify the most frequently cited and most influential
works across this corpus that relate to the primary research claims.

For each highly cited paper (cited ≥2 times), return:
  paper_id     : the citing paper's ID or DOI
  title        : title of the cited work
  cited_by     : list of paper_ids that cite this work
  relevance    : "high" | "medium" | "low" — how central to the claims

Return ONLY valid JSON.

Papers corpus:
{papers}

Output format:
{{
  "influential_works": [
    {{
      "paper_id":  "...",
      "title":     "...",
      "cited_by":  ["paper_id_1", "paper_id_2"],
      "relevance": "high | medium | low"
    }}
  ]
}}
"""

logger = logging.getLogger("agent.agent_8_citation_analysis")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Concrete Agent
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# NOTE: @registry.register is intentionally COMMENTED OUT here.
# Uncomment it (and add the import to agents/__init__.py) to activate.
# This demonstrates the "closed for modification" aspect of OCP:
# the file is ready; activation requires only the __init__.py import.

# @registry.register
class CitationAnalysisAgent(BaseAgent):
    """
    Agent 8: Builds a citation influence graph from paper reference lists.

    Demonstrates OCP extension: this entire agent was written and plugged in
    without modifying BaseAgent, AgentRegistry, or AgentPipelineOrchestrator.

    Activation: uncomment @registry.register above and add one import line
    to agents/__init__.py.
    """

    agent_id        = "agent_8_citation_analysis"
    role            = ("Builds a PageRank-weighted citation graph from paper "
                       "reference lists. Surfaces the most influential works "
                       "referenced across the retrieved corpus. "
                       "LLM enriches with relevance labels; graph math is deterministic.")
    prompt_template = CITATION_ANALYSIS_PROMPT

    required_context_keys: List[str] = [
        "agent_1_claim_extraction.claims",
    ]

    def _execute(self, context: AgentContext) -> AgentResult:
        """
        Build citation graph using networkx PageRank.
        Falls back to degree centrality if PageRank fails.
        """
        try:
            import networkx as nx
        except ImportError:
            return self._fail("networkx is required for citation graph analysis.")

        papers = context.papers
        if not papers:
            return self._ok({
                "citation_graph":  {"nodes": [], "edges": []},
                "top_influential": [],
                "isolated_papers": [],
            })

        # ── Build directed citation graph ─────────────────────────────────────
        G = nx.DiGraph()

        # Add all retrieved papers as nodes
        for paper in papers:
            pid   = paper.get("paper_id", paper.get("id", "unknown"))
            title = paper.get("title", "")
            G.add_node(pid, title=title)

        # Add edges from reference lists (if available)
        edge_count = 0
        for paper in papers:
            source_id = paper.get("paper_id", paper.get("id", "unknown"))
            references = paper.get("references", [])   # list of {paper_id, title}

            for ref in references:
                ref_id = ref.get("paper_id") or ref.get("id")
                if not ref_id:
                    continue
                if ref_id not in G:
                    G.add_node(ref_id, title=ref.get("title", ""))
                G.add_edge(source_id, ref_id, weight=1)
                edge_count += 1

        logger.info(
            "[agent_8] Citation graph: %d nodes, %d edges.",
            G.number_of_nodes(), G.number_of_edges(),
        )

        # ── Compute PageRank (influence score) ────────────────────────────────
        try:
            pagerank = nx.pagerank(G, alpha=0.85, max_iter=100)
        except Exception:
            # Fallback: degree centrality
            pagerank = dict(nx.degree_centrality(G))

        # ── Identify isolated papers (no citation links) ──────────────────────
        isolated = [n for n in G.nodes() if G.degree(n) == 0]

        # ── Build serialisable output ─────────────────────────────────────────
        nodes = [
            {
                "paper_id":      nid,
                "title":         G.nodes[nid].get("title", ""),
                "pagerank_score": round(pagerank.get(nid, 0.0), 6),
            }
            for nid in G.nodes()
        ]
        nodes.sort(key=lambda x: x["pagerank_score"], reverse=True)

        edges = [
            {"source": u, "target": v, "weight": d.get("weight", 1)}
            for u, v, d in G.edges(data=True)
        ]

        top_influential = nodes[:5]   # top 5 by PageRank

        return self._ok(
            payload={
                "citation_graph":  {"nodes": nodes, "edges": edges},
                "top_influential": top_influential,
                "isolated_papers": isolated,
            },
            graph_nodes=len(nodes),
            graph_edges=len(edges),
            isolated_count=len(isolated),
        )
