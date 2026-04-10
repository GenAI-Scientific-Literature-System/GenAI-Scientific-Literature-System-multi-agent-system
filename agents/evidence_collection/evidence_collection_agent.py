"""
agents/evidence_collection/evidence_collection_agent.py
Derived from GenAI Agent 2 (src/agents/agent2_evidence.py)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GENAI FORMAL MODEL: Evidence = {spans[], strength, claim_id}
RAG: builds a combined claim query from up to 4 claims, retrieves top-3 chunks.
[V2] verify_evidence_spans() — confirms every span physically exists in source text.
"""
from __future__ import annotations
import logging
from typing import Any, Dict, List
from agents.base.base_agent import AgentContext, AgentResult, BaseAgent
from core.registry.agent_registry import registry

EVIDENCE_PROMPT = """\
You are Agent 2 — a scientific evidence attribution specialist.
Derived from GenAI Agent 2 (src/agents/agent2_evidence.py).

For each claim, locate the strongest supporting evidence spans in the paper text.

Rules:
  - Spans MUST be exact substrings of the text (V2 guard verifies this)
  - Do NOT fabricate evidence
  - Return ONLY valid JSON

Claims: {claims}
Paper text: {text}

Output format:
{{
  "evidence": [
    {{"claim_id": 0, "spans": ["<exact span>"], "strength": "high|medium|low", "justification": "..."}}
  ]
}}
"""

logger = logging.getLogger("agent.agent_2_evidence_collection")


@registry.register
class EvidenceCollectionAgent(BaseAgent):
    """
    Agent 2: Attributes verbatim evidence spans to each claim.
    Derived from GenAI Agent 2 (src/agents/agent2_evidence.py)
    """
    agent_id        = "agent_2_evidence_collection"
    role            = (
        "Attributes verbatim evidence spans from source texts to each claim. "
        "Uses RAG: combined claim query → top-3 chunks (~900 chars). "
        "Applies V2 Anti-Hallucination: verify_evidence_spans() substring check."
    )
    prompt_template = EVIDENCE_PROMPT
    genai_origin    = "src/agents/agent2_evidence.py"

    required_context_keys: List[str] = ["agent_1_claim_extraction.claims"]

    def __init__(self, retriever=None) -> None:
        super().__init__()
        self._retriever = retriever

    def _execute(self, context: AgentContext) -> AgentResult:
        try:
            from genai_system.src.agents.agent2_evidence import attribute_evidence
            from genai_system.src.models.schemas import Claim
        except ImportError as exc:
            return self._fail(f"[Derived from GenAI Agent 2] Could not import attribute_evidence: {exc}")

        raw_claims: List[Dict[str, Any]] = context.upstream("agent_1_claim_extraction", "claims", default=[])
        if not raw_claims:
            return self._ok({"claims": [], "spans_removed": 0, "per_claim_spans": {}})

        try:
            claims_objs = [
                Claim(**{k: v for k, v in c.items() if k in Claim.__dataclass_fields__})
                for c in raw_claims
            ]
        except Exception as exc:
            return self._fail(f"Claim reconstruction failed: {exc}")

        combined_text = "\n\n".join(
            p.get("text", p.get("abstract", "")) for p in context.papers
        )
        total_removed = 0
        per_claim_spans: Dict[str, int] = {}

        try:
            enriched_claims, removed = attribute_evidence(
                claims=claims_objs, text=combined_text, retriever=self._retriever
            )
            total_removed = removed
            for c in enriched_claims:
                per_claim_spans[c.id] = len(getattr(c, "evidence_spans", []))
            serialised = [c.to_dict() if hasattr(c, "to_dict") else vars(c) for c in enriched_claims]
        except Exception as exc:
            logger.error("[agent_2] attribute_evidence raised: %s", exc)
            serialised    = raw_claims
            total_removed = 0

        logger.info("[agent_2] V2 removed %d fabricated spans.", total_removed)
        return self._ok(
            payload={"claims": serialised, "spans_removed": total_removed, "per_claim_spans": per_claim_spans},
            spans_removed=total_removed,
        )
