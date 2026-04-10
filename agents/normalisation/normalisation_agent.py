"""
agents/normalisation/normalisation_agent.py
Derived from GenAI Agent 3 (src/agents/agent3_normalize.py)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GENAI FORMAL MODEL: PREDICATE_MAP × DOMAIN_MAP → canonical forms
Rule-based regex mapping — zero LLM tokens, fully reproducible.
  PREDICATE_MAP: outperform/surpass → "outperforms"; improve/enhance → "improves"; etc.
  DOMAIN_MAP:    nlp/natural language → "NLP"; vision/image → "CV"; etc.
"""
from __future__ import annotations
import logging
from typing import Any, Dict, List
from agents.base.base_agent import AgentContext, AgentResult, BaseAgent
from core.registry.agent_registry import registry

logger = logging.getLogger("agent.agent_3_normalisation")


@registry.register
class NormalisationAgent(BaseAgent):
    """
    Agent 3: Standardises claim predicates and domains. Zero LLM tokens.
    Derived from GenAI Agent 3 (src/agents/agent3_normalize.py)
    """
    agent_id        = "agent_3_normalisation"
    role            = (
        "Standardises claim predicates (e.g. 'surpass','beat' → 'outperforms') "
        "and domains (e.g. 'natural language','text' → 'NLP') using GenAI's "
        "PREDICATE_MAP and DOMAIN_MAP regex tables. Deterministic — zero LLM tokens."
    )
    prompt_template = None
    genai_origin    = "src/agents/agent3_normalize.py"

    required_context_keys: List[str] = ["agent_2_evidence_collection.claims"]

    def _execute(self, context: AgentContext) -> AgentResult:
        try:
            from genai_system.src.agents.agent3_normalize import normalise_claims
            from genai_system.src.models.schemas import Claim
        except ImportError as exc:
            return self._fail(f"[Derived from GenAI Agent 3] Could not import normalise_claims: {exc}")

        raw_claims: List[Dict[str, Any]] = context.upstream(
            "agent_2_evidence_collection", "claims", default=[]
        )
        if not raw_claims:
            return self._ok({"claims": [], "normalised": 0})

        try:
            claims_objs = [
                Claim(**{k: v for k, v in c.items() if k in Claim.__dataclass_fields__})
                for c in raw_claims
            ]
        except Exception as exc:
            return self._fail(f"Claim reconstruction failed: {exc}")

        before_predicates = [c.predicate for c in claims_objs]
        before_domains    = [c.domain    for c in claims_objs]

        normalised_claims = normalise_claims(claims_objs)

        n_changed = sum(
            1 for i, c in enumerate(normalised_claims)
            if c.predicate != before_predicates[i] or c.domain != before_domains[i]
        )
        serialised = [
            c.to_dict() if hasattr(c, "to_dict") else vars(c)
            for c in normalised_claims
        ]
        logger.info("[agent_3] Normalised %d of %d claims.", n_changed, len(serialised))
        return self._ok(
            payload={"claims": serialised, "normalised": n_changed},
            fields_changed=n_changed,
        )
