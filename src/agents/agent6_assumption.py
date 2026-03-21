"""
Agent 6: Assumption Extraction  A = (type, scope, constraint, explicitness, evidence_span)

UPGRADE: retriever fetches assumption-rich sections (limitations, setup, constraints)
         instead of sending the full text. Token reduction: ~75 %.
"""
import logging
from typing import List, Optional
from src.models.schemas import Claim, Assumption, AssumptionType
from src.mistral_client import call_mistral, sanitize_for_prompt
from config import ASSUMPTION_PROMPT

logger = logging.getLogger(__name__)

_ASSUMPTION_QUERY = "assumptions limitations constraints requirements scope experimental setup dataset"


def extract_assumptions(
    text: str,
    claims: List[Claim],
    retriever=None,              # DocumentRetriever | None
) -> List[Assumption]:

    if retriever is not None:
        context = retriever.retrieve(_ASSUMPTION_QUERY, top_k=4)
        context = sanitize_for_prompt(context, max_chars=900)
        logger.debug("Agent 6 [RAG]: %d chars of retrieved context.", len(context))
    else:
        context = sanitize_for_prompt(text, max_chars=1600)

    prompt = ASSUMPTION_PROMPT.format(text=context)

    result = call_mistral(
        prompt,
        system="Return only a JSON object with an 'assumptions' array. No prose.",
        max_tokens=500,
    )

    assumptions: List[Assumption] = []
    if not result:
        logger.warning("Agent 6: No result from Mistral.")
        return assumptions

    raw_list = result.get("assumptions", result) if isinstance(result, dict) else result
    if not isinstance(raw_list, list):
        raw_list = []

    for item in raw_list:
        if not isinstance(item, dict):
            continue
        a = Assumption(
            type=item.get("type", AssumptionType.DOMAIN),
            constraint=str(item.get("constraint", "")).strip(),
            explicit=bool(item.get("explicit", True)),
            span=str(item.get("span", "")).strip(),
        )
        if a.constraint:
            assumptions.append(a)

    logger.info("Agent 6: Extracted %d assumptions.", len(assumptions))
    return assumptions


def assign_assumptions_to_claims(claims: List[Claim], assumptions: List[Assumption]) -> List[Claim]:
    """Keyword-overlap matching — zero LLM tokens."""
    import re
    for claim in claims:
        claim_words = set(re.split(r'\W+', f"{claim.subject} {claim.object} {claim.method}".lower()))
        for assumption in assumptions:
            constraint_words = set(w for w in re.split(r'\W+', assumption.constraint.lower()) if len(w) > 3)
            if len(constraint_words & claim_words) >= 1:
                claim.assumptions.append(assumption)
    return claims
