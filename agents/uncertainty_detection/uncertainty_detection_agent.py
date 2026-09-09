"""
agents/uncertainty_detection/uncertainty_detection_agent.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Agent 5: Uncertainty Propagation & Research Gap Detection

Derived from GenAI Agent 5 (src/agents/agent5_uncertainty.py)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
IDENTITY
  agent_id        : "agent_5_uncertainty_detection"
  genai_origin    : src/agents/agent5_uncertainty.py

GENAI FORMAL MODELS (preserved exactly)

  Uncertainty per claim (from GenAI config weights):
    u(claim) = CONFLICT_WEIGHT   * (contradictions / total_relations)   [0.30]
             + EVIDENCE_WEIGHT   * EVIDENCE_STRENGTH_MAP[strength]      [0.30]
             + STABILITY_WEIGHT  * (WEAK_assumptions / total_assumps)   [0.40]

  EDG propagation (20% bleed to neighbours):
    u[node] = 0.8 * u[node] + 0.2 * avg(u[neighbours])

  Multi-signal gap score (deterministic graph analysis):
    gap_score(node) = 0.30 * low_degree(node)       # poorly connected
                    + 0.30 * high_uncertainty         # uncertain claim
                    + 0.20 * low_evidence             # weak evidence
                    + 0.20 * low_centrality           # not a bridge node
    gaps = top-N by gap_score; LLM writes only a gap label (~35 tokens)

ANTI-HALLUCINATION
  [V4] filter_hallucinated_gaps() — verifies gap text references actual claim content.

UNIFIED ARCHITECTURE MODULES USED
  - src/graph/edg.py  → EpistemicDependencyGraph (degree, betweenness centrality)
  - src/reasoning.py  → NetworkX betweenness_centrality computation

PIPELINE CONTRACT
  Input  :
    "agent_3_normalisation"       → "claims"
    "agent_4_agreement_detection" → "agreements"
  Output :
    {
      "gaps":          List[ResearchGap dict],
      "uncertainties": {claim_id: float},
      "edg_stats":     {"nodes": int, "edges": int, "top_gap_score": float}
    }
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from agents.base.base_agent import AgentContext, AgentResult, BaseAgent
from core.registry.agent_registry import registry

logger = logging.getLogger("agent.agent_5_uncertainty_detection")

# ── Prompt: derived from GenAI Agent 5's _GAP_LABEL_SYSTEM (preserved) ────────
# Original: "Write ONE sentence (max 20 words) naming the research gap for this
# claim based on its graph analysis scores."
# Expanded to full template format (Unified architecture pattern).
GAP_LABEL_PROMPT = """\
You are Agent 5 — a scientific research gap analyst.
Derived from GenAI Agent 5 (src/agents/agent5_uncertainty.py).

A claim node in the Epistemic Dependency Graph has been flagged as a
research gap via the multi-signal gap score formula:
  gap_score = 0.30*low_degree + 0.30*high_uncertainty + 0.20*low_evidence + 0.20*low_centrality

Claim      : {claim_text}
Domain     : {domain}
Gap score  : {gap_score:.3f}
Degree     : {degree}        (graph connectivity)
Betweenness: {betweenness:.3f}  (bridge importance)
Uncertainty: {uncertainty:.3f}  (propagated uncertainty)

Write ONE sentence (≤ 20 words) naming the specific research gap.
Be precise about what knowledge is missing or under-studied.

Return ONLY: {{"gap": "...", "type": "theoretical|empirical|methodological", "priority": "high|medium|low"}}
"""


@registry.register
class UncertaintyDetectionAgent(BaseAgent):
    """
    Agent 5: Propagates uncertainty through the EDG and detects research gaps.

    Derived from GenAI Agent 5 (src/agents/agent5_uncertainty.py)
    Wraps:
      genai_system.src.agents.agent5_uncertainty.propagate_uncertainty()
      genai_system.src.agents.agent5_uncertainty.detect_gaps()

    The full multi-signal gap scoring formula and EDG propagation logic from
    the original GenAI codebase are preserved inside those functions.
    This class adds only the BaseAgent interface and multi-paper aggregation.
    """

    agent_id        = "agent_5_uncertainty_detection"
    role            = (
        "Propagates uncertainty scores through the Epistemic Dependency Graph (EDG): "
        "u[node] = 0.8*local + 0.2*avg(neighbours). "
        "Detects research gaps using multi-signal scoring: "
        "gap_score = 0.30*low_degree + 0.30*high_u + 0.20*low_evidence + 0.20*low_centrality. "
        "LLM writes only gap label strings (~35 tokens). "
        "Applies V4 Anti-Hallucination: filter_hallucinated_gaps()."
    )
    prompt_template = GAP_LABEL_PROMPT
    genai_origin    = "src/agents/agent5_uncertainty.py"   # GenAI credit

    required_context_keys: List[str] = [
        "agent_3_normalisation.claims",
        "agent_4_agreement_detection.agreements",
    ]

    def _execute(self, context: AgentContext) -> AgentResult:
        """
        Runs GenAI Agent 5's two-phase pipeline:
          Phase 1 — propagate_uncertainty(): computes and propagates u(claim)
          Phase 2 — detect_gaps():           multi-signal gap score + LLM labels

        Both functions are imported from genai_system unchanged (OCP).
        """
        try:
            from genai_system.src.agents.agent5_uncertainty import (
                propagate_uncertainty,
                detect_gaps,
                compute_uncertainty,
            )
            from genai_system.src.models.schemas import Claim, Agreement
            from genai_system.src.graph.edg import EpistemicDependencyGraph
        except ImportError as exc:
            return self._fail(
                f"[Derived from GenAI Agent 5] Could not import agent5_uncertainty: {exc}"
            )

        raw_claims: List[Dict] = context.upstream("agent_3_normalisation", "claims", default=[])
        raw_agreements: List[Dict] = context.upstream(
            "agent_4_agreement_detection", "agreements", default=[]
        )

        if not raw_claims:
            return self._ok({
                "gaps": [], "uncertainties": {},
                "edg_stats": {"nodes": 0, "edges": 0, "top_gap_score": 0.0},
            })

        # Reconstruct typed objects from serialised pipeline dicts
        try:
            claims_objs = [
                Claim(**{k: v for k, v in c.items() if k in Claim.__dataclass_fields__})
                for c in raw_claims
            ]
            agreement_objs = [
                Agreement(**{k: v for k, v in a.items() if k in Agreement.__dataclass_fields__})
                for a in raw_agreements
            ]
        except Exception as exc:
            return self._fail(f"Object reconstruction failed: {exc}")

        # ── Phase 1: Uncertainty propagation (GenAI Agent 5, Phase 1) ──────────
        # propagate_uncertainty() runs:
        #   - compute_uncertainty() per claim (conflict + evidence + stability weights)
        #   - EDG neighbourhood bleed (20%)
        try:
            claims_objs = propagate_uncertainty(claims_objs, agreement_objs)
        except Exception as exc:
            logger.warning("[agent_5] propagate_uncertainty failed: %s — using local only.", exc)
            for c in claims_objs:
                c.uncertainty = compute_uncertainty(c, agreement_objs)

        uncertainties: Dict[str, float] = {c.id: c.uncertainty for c in claims_objs}

        # ── Phase 2: Gap detection (GenAI Agent 5, Phase 2) ─────────────────────
        # detect_gaps() runs:
        #   - gap_score() for every node (multi-signal formula)
        #   - Top-N selection (max 6, min 2)
        #   - LLM label generation (_label_gap, ~35 tokens each)
        #   - V4 guard: filter_hallucinated_gaps()
        try:
            edg = EpistemicDependencyGraph()
            # Populate EDG with agreement edges so graph metrics are meaningful
            for ag in agreement_objs:
                if getattr(ag, "relation", "") != "unrelated":
                    edg.add_relation(
                        ag.claim_i_id, ag.claim_j_id,
                        ag.relation,
                        getattr(ag, "confidence", 0.5),
                        getattr(ag, "reason", ""),
                    )
            gaps, dropped = detect_gaps(claims_objs, edg)
        except Exception as exc:
            logger.error("[agent_5] detect_gaps raised: %s", exc)
            gaps    = []
            dropped = 0

        serialised_gaps = [
            g.to_dict() if hasattr(g, "to_dict") else vars(g)
            for g in gaps
        ]

        # EDG statistics for pipeline_summary
        try:
            edg_nodes = edg.G.number_of_nodes() if hasattr(edg, "G") else len(claims_objs)
            edg_edges = edg.G.number_of_edges() if hasattr(edg, "G") else len(agreement_objs)
        except Exception:
            edg_nodes, edg_edges = len(claims_objs), len(agreement_objs)

        top_gap_score = 0.0
        if serialised_gaps:
            top_gap_score = serialised_gaps[0].get("gap_signals", {}).get("gap_score", 0.0)

        logger.info(
            "[agent_5] %d gaps detected. EDG: %d nodes, %d edges. top_gap=%.3f",
            len(serialised_gaps), edg_nodes, edg_edges, top_gap_score,
        )

        return self._ok(
            payload={
                "gaps":          serialised_gaps,
                "uncertainties": uncertainties,
                "edg_stats":     {
                    "nodes":         edg_nodes,
                    "edges":         edg_edges,
                    "top_gap_score": round(top_gap_score, 3),
                },
            },
            gap_count=len(serialised_gaps),
            v4_drops=dropped,
            avg_uncertainty=(
                round(sum(uncertainties.values()) / len(uncertainties), 3)
                if uncertainties else 0.0
            ),
        )
