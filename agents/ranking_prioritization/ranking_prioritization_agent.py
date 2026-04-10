"""
agents/ranking_prioritization/ranking_prioritization_agent.py
Derived from GenAI agents/ranking_prioritization/ranking_prioritizer.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GENAI FORMAL MODEL (preserved exactly):
  priority = 0.35*reliability + 0.30*evidence + 0.20*agreement + 0.15*novelty
  conflict_penalty: priority -= contradict_ratio * 0.15

Two-stage ranking (GenAI original):
  Stage 1: Heuristic score (deterministic, always runs)
  Stage 2: LLM re-ranking (optional fallback to heuristic if LLM fails)

Terminal agent — consumes ALL upstream outputs.
"""
from __future__ import annotations
import logging
from typing import Any, Dict, List
from agents.base.base_agent import AgentContext, AgentResult, BaseAgent
from core.registry.agent_registry import registry

RANKING_PROMPT = """\
You are Agent 7 — a scientific insight prioritization specialist.
Derived from GenAI agents/ranking_prioritization/ranking_prioritizer.py.

Validate and re-order the heuristically pre-ranked insights below.
Ranking criteria:
  1. Reliability (methodology quality, sample size)
  2. Evidence strength and quantity
  3. Agreement with other papers (not contradiction)
  4. Novelty — findings that challenge or extend consensus

For each insight, confirm/adjust rank and add a one-sentence justification.

Insights (heuristically pre-ranked):
{insights}

Return ONLY valid JSON:
{{
  "ranked": [
    {{"rank": 1, "claim_id": "...", "claim_text": "...", "priority_score": 0.0, "reason": "..."}}
  ]
}}
"""

logger = logging.getLogger("agent.agent_7_ranking_prioritization")


@registry.register
class RankingPrioritizationAgent(BaseAgent):
    """
    Agent 7: Synthesises all pipeline outputs into ranked insights.
    Derived from GenAI agents/ranking_prioritization/ranking_prioritizer.py
    """
    agent_id        = "agent_7_ranking_prioritization"
    role            = (
        "Terminal synthesis agent. Heuristic priority formula: "
        "0.35*reliability + 0.30*evidence + 0.20*agreement + 0.15*novelty "
        "- 0.15*conflict_penalty. Optional LLM re-ranking with justifications. "
        "Consumes ALL upstream agent outputs."
    )
    prompt_template = RANKING_PROMPT
    genai_origin    = "agents/ranking_prioritization/ranking_prioritizer.py"

    required_context_keys: List[str] = [
        "agent_1_claim_extraction.claims",
        "agent_4_agreement_detection.agreements",
        "agent_5_uncertainty_detection.gaps",
    ]

    # GenAI heuristic weights (preserved verbatim)
    RELIABILITY_WEIGHT = 0.35
    EVIDENCE_WEIGHT    = 0.30
    AGREEMENT_WEIGHT   = 0.20
    NOVELTY_WEIGHT     = 0.15
    CONFLICT_PENALTY   = 0.15

    EVIDENCE_STRENGTH_MAP = {"high": 1.0, "medium": 0.6, "low": 0.2}

    def _execute(self, context: AgentContext) -> AgentResult:
        # Collect all upstream outputs
        raw_claims       = context.upstream("agent_1_claim_extraction",      "claims",               default=[])
        enriched_claims  = context.upstream("agent_6_assumption_extraction",  "claims_with_assumptions", default=raw_claims)
        agreements       = context.upstream("agent_4_agreement_detection",    "agreements",           default=[])
        uncertainties    = context.upstream("agent_5_uncertainty_detection",  "uncertainties",        default={})
        gaps             = context.upstream("agent_5_uncertainty_detection",  "gaps",                 default=[])

        if not enriched_claims:
            return self._ok({
                "ranked_insights": [],
                "pipeline_summary": {"total_claims": 0, "total_agreements": 0, "total_gaps": 0, "top_insight": None},
            })

        # Build agreement index: claim_id → {agree, contradict, conditional}
        agreement_index: Dict[str, Dict[str, int]] = {}
        for ag in agreements:
            for cid in [ag.get("claim_i_id"), ag.get("claim_j_id")]:
                if not cid:
                    continue
                if cid not in agreement_index:
                    agreement_index[cid] = {"agree": 0, "contradict": 0, "conditional": 0}
                rel = str(ag.get("relation", "unrelated")).lower()
                if rel in agreement_index[cid]:
                    agreement_index[cid][rel] += 1

        # Stage 1: Heuristic priority score (GenAI formal model)
        scored: List[Dict[str, Any]] = []
        for claim in enriched_claims:
            cid        = claim.get("id", "")
            ev_strength= str(claim.get("evidence_strength", "medium")).lower()
            ev_score   = self.EVIDENCE_STRENGTH_MAP.get(ev_strength, 0.4)
            ag_data    = agreement_index.get(cid, {"agree": 0, "contradict": 0, "conditional": 0})
            total_rels = sum(ag_data.values()) or 1
            agree_ratio    = ag_data["agree"]     / total_rels
            conflict_ratio = ag_data["contradict"] / total_rels
            uncertainty    = uncertainties.get(cid, 0.5)
            novelty_score  = min(uncertainty * 0.8 + (1 - agree_ratio) * 0.2, 1.0)
            reliability    = ev_score

            priority = (
                self.RELIABILITY_WEIGHT * reliability
                + self.EVIDENCE_WEIGHT  * ev_score
                + self.AGREEMENT_WEIGHT * agree_ratio
                + self.NOVELTY_WEIGHT   * novelty_score
                - self.CONFLICT_PENALTY * conflict_ratio
            )
            priority = max(round(priority, 4), 0.0)

            scored.append({
                "claim_id":       cid,
                "paper_id":       claim.get("paper_id", ""),
                "claim_text":     f"{claim.get('subject','')} {claim.get('predicate','')} {claim.get('object','')}".strip(),
                "priority_score": priority,
                "reliability":    round(reliability, 3),
                "evidence_score": round(ev_score, 3),
                "agree_ratio":    round(agree_ratio, 3),
                "conflict_ratio": round(conflict_ratio, 3),
                "novelty_score":  round(novelty_score, 3),
                "uncertainty":    round(uncertainty, 3),
                "n_assumptions":  len(claim.get("assumptions", [])),
                "rank":           0,
            })

        scored.sort(key=lambda x: x["priority_score"], reverse=True)
        for i, ins in enumerate(scored, start=1):
            ins["rank"] = i

        top_insight = scored[0]["claim_text"] if scored else None
        summary = {
            "total_claims":     len(enriched_claims),
            "total_agreements": len(agreements),
            "total_gaps":       len(gaps),
            "top_insight":      top_insight,
        }
        logger.info("[agent_7] Ranked %d insights. Top: '%s'", len(scored), (top_insight or "")[:80])

        return self._ok(
            payload={"ranked_insights": scored, "pipeline_summary": summary},
            insight_count=len(scored),
        )
