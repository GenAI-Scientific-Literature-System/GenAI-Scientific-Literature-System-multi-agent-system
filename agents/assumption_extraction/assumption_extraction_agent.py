"""
agents/assumption_extraction/assumption_extraction_agent.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Agent 6 (pipeline position 3): Assumption Extraction & Assignment

Derived from GenAI Agent 6 (src/agents/agent6_assumption.py)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
IDENTITY
  agent_id        : "agent_6_assumption_extraction"
  pipeline_position: 3 (runs BEFORE normalisation and agreement detection,
                     so assumptions are available for Agent 4's formal function)
  genai_origin    : src/agents/agent6_assumption.py

GENAI FORMAL MODEL (preserved exactly)
  A = (type, scope, constraint, explicitness, evidence_span)
  RAG query: "assumptions limitations constraints requirements scope experimental setup dataset"
  Token reduction vs full text: ~75% when RAG retriever is provided.

  Two-phase execution (both from GenAI original):
    Phase 1: LLM extract_assumptions()          — Mistral, targeted RAG context
    Phase 2: assign_assumptions_to_claims()     — keyword-overlap, ZERO tokens
    Phase V5: verify_all_assumptions()           — 4-tier NLI guard (agent6_1_verify.py)

ANTI-HALLUCINATION
  [V5] deep_assumption_ground() via agent6_1_verify.py:
    Tier 1: exact span match         → VERIFIED (score=1.0)
    Tier 2: bigram overlap ≥ 0.50   → VERIFIED
    Tier 3: unigram overlap ≥ 0.40  → WEAK
    Tier 4: local NLI (implicit)    → upgrade/downgrade
    Else:                            → REJECTED (filtered out)
    Zero Mistral tokens.

PIPELINE CONTRACT
  Input  :
    "agent_2_evidence_collection" → "claims"
    context.papers                → source texts
  Output :
    {
      "assumptions":            List[Assumption dict],
      "claims_with_assumptions": List[Claim dict],
      "assignment_stats":       {claim_id: n_assumptions}
    }
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from agents.base.base_agent import AgentContext, AgentResult, BaseAgent
from core.registry.agent_registry import registry

logger = logging.getLogger("agent.agent_6_assumption_extraction")

# ── Prompt: derived from GenAI ASSUMPTION_PROMPT (config.py) ─────────────────
# Original system: "Return only a JSON object with an 'assumptions' array. No prose."
# Expanded to full ownable template (Unified architecture pattern).
ASSUMPTION_PROMPT = """\
You are Agent 6 — a scientific assumption extraction specialist.
Derived from GenAI Agent 6 (src/agents/agent6_assumption.py).

Formal assumption model: A = (type, scope, constraint, explicitness, evidence_span)

Given a research paper text, extract ALL underlying assumptions the authors make
(explicit or implicit). Focus on:
  - Methodological assumptions (experimental design, controls, baselines)
  - Statistical assumptions (normality, independence, sample size)
  - Domain assumptions (population scope, generalisability limits)
  - Scope assumptions (what the paper claims NOT to address)

For each assumption return:
  text         : the assumption in one clear sentence (this is the "constraint" field)
  type         : "methodological" | "statistical" | "domain" | "scope"
  scope        : the section or context where this assumption applies
  explicitness : "EXPLICIT" if stated directly, "IMPLICIT" if inferred
  span         : verbatim text span (≤30 words) from the paper that indicates this assumption

Rules:
  - Every assumption MUST be grounded in the paper text (V5 NLI guard checks this)
  - IMPLICIT assumptions must still be inferrable from the text
  - Return ONLY valid JSON — no prose outside the object

Paper text:
{text}

Output format:
{{
  "assumptions": [
    {{
      "text":         "...",
      "type":         "methodological | statistical | domain | scope",
      "scope":        "...",
      "explicitness": "EXPLICIT | IMPLICIT",
      "span":         "..."
    }}
  ]
}}
"""

# RAG query used by GenAI Agent 6 (preserved verbatim)
_ASSUMPTION_QUERY = (
    "assumptions limitations constraints requirements scope experimental setup dataset"
)


@registry.register
class AssumptionExtractionAgent(BaseAgent):
    """
    Agent 6 (pipeline position 3): Extracts and assigns assumptions to claims.

    Derived from GenAI Agent 6 (src/agents/agent6_assumption.py)
    Wraps:
      genai_system.src.agents.agent6_assumption.extract_assumptions()
      genai_system.src.agents.agent6_assumption.assign_assumptions_to_claims()
      genai_system.src.agents.agent6_1_verify.verify_all_assumptions()  [V5]

    Runs early in the pipeline (position 3, before Agent 4) because Agent 4's
    formal agreement function R(Ci,Cj|Ai,Aj) requires assumption-enriched claims.
    """

    agent_id        = "agent_6_assumption_extraction"
    role            = (
        "Extracts explicit and implicit assumptions A=(type, scope, constraint, "
        "explicitness, evidence_span) from paper texts using Mistral with RAG "
        "(~75% token reduction). Assigns assumptions to claims via keyword-overlap "
        "(zero extra tokens). These enriched claims feed Agent 4's formal function "
        "R(Ci, Cj | Ai, Aj). Applies V5 Anti-Hallucination: four-tier NLI guard."
    )
    prompt_template = ASSUMPTION_PROMPT
    genai_origin    = "src/agents/agent6_assumption.py"   # GenAI credit

    required_context_keys: List[str] = [
        "agent_2_evidence_collection.claims",
    ]

    def __init__(self, retriever=None) -> None:
        super().__init__()
        self._retriever = retriever

    def _execute(self, context: AgentContext) -> AgentResult:
        """
        Two-phase GenAI Agent 6 logic:
          Phase 1: extract_assumptions() — LLM with RAG context
          Phase 2: assign_assumptions_to_claims() — deterministic keyword overlap
          Phase V5: verify_all_assumptions() — four-tier NLI guard (agent6_1_verify)
        """
        try:
            from genai_system.src.agents.agent6_assumption import (
                extract_assumptions,
                assign_assumptions_to_claims,
            )
            from genai_system.src.models.schemas import Claim
        except ImportError as exc:
            return self._fail(
                f"[Derived from GenAI Agent 6] Could not import agent6_assumption: {exc}"
            )

        raw_claims: List[Dict] = context.upstream(
            "agent_2_evidence_collection", "claims", default=[]
        )
        if not raw_claims:
            return self._ok({
                "assumptions": [],
                "claims_with_assumptions": [],
                "assignment_stats": {},
            })

        try:
            claims_objs = [
                Claim(**{k: v for k, v in c.items() if k in Claim.__dataclass_fields__})
                for c in raw_claims
            ]
        except Exception as exc:
            return self._fail(f"Claim reconstruction failed: {exc}")

        all_assumptions = []
        for paper in context.papers:
            paper_id = paper.get("paper_id", paper.get("id", "unknown"))
            text     = paper.get("text", paper.get("abstract", ""))
            if not text:
                continue
            try:
                # Phase 1: LLM extraction with targeted RAG context
                paper_assumptions = extract_assumptions(
                    text=text,
                    claims=claims_objs,
                    retriever=self._retriever,
                )
                # Phase V5: four-tier NLI verification (BaseAgent hook)
                paper_assumptions = self._assumption_verify(paper_assumptions, text)

                all_assumptions.extend(paper_assumptions)
                logger.info("[agent_6] Paper '%s': %d assumptions after V5.", paper_id, len(paper_assumptions))
            except Exception as exc:
                logger.error("[agent_6] extract_assumptions failed for '%s': %s", paper_id, exc)

        # Phase 2: keyword-overlap assignment — zero extra LLM tokens
        claims_with_assumptions = assign_assumptions_to_claims(claims_objs, all_assumptions)

        assignment_stats = {
            c.id: len(getattr(c, "assumptions", []))
            for c in claims_with_assumptions
        }

        serialised_assumptions = [
            a.to_dict() if hasattr(a, "to_dict") else vars(a)
            for a in all_assumptions
        ]
        serialised_claims = [
            c.to_dict() if hasattr(c, "to_dict") else vars(c)
            for c in claims_with_assumptions
        ]

        logger.info(
            "[agent_6] %d assumptions extracted, %d claims enriched.",
            len(serialised_assumptions), len(serialised_claims),
        )

        return self._ok(
            payload={
                "assumptions":             serialised_assumptions,
                "claims_with_assumptions": serialised_claims,
                "assignment_stats":        assignment_stats,
            },
            assumption_count=len(serialised_assumptions),
            rag_active=self._retriever is not None,
        )
