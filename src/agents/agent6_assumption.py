"""
Agent 6: Assumption Extraction  A = (type, scope, constraint, explicitness, evidence_span)

OCP FIX: Prompt returns {text, type, scope, explicitness} but parser was reading
         {constraint, span, explicit} — field names never matched, so 0 assumptions
         were ever parsed. Parser now maps both naming conventions.

UPGRADE: retriever fetches assumption-rich sections (limitations, setup, constraints)
         instead of sending the full text. Token reduction: ~75 %.
"""
import logging
from typing import List
from src.models.schemas import Claim, Assumption, AssumptionType, VerificationStatus
from src.mistral_client import call_mistral, sanitize_for_prompt
from config import ASSUMPTION_PROMPT

logger = logging.getLogger(__name__)

_ASSUMPTION_QUERY = "assumptions limitations constraints requirements scope experimental setup dataset"

# Map the prompt's type strings → AssumptionType enum values
_TYPE_MAP = {
    "methodological": AssumptionType.METHOD,
    "method":         AssumptionType.METHOD,
    "statistical":    AssumptionType.STATISTICAL,
    "domain":         AssumptionType.DOMAIN,
    "implicit":       AssumptionType.DOMAIN,   # no IMPLICIT enum value — fall back to DOMAIN
    "scope":          AssumptionType.SCOPE,
}


def _parse_assumption(item: dict) -> Assumption | None:
    """
    Parse one assumption dict from the LLM response.
    Accepts both prompt-side names (text/scope/explicitness) and
    schema-side names (constraint/span/explicit) so either prompt
    wording works without changing the prompt.
    """
    if not isinstance(item, dict):
        return None

    # ── constraint: try "text" first (what the prompt asks for), then "constraint"
    constraint = str(item.get("text") or item.get("constraint") or "").strip()
    if not constraint:
        return None

    # ── span / scope
    span = str(item.get("scope") or item.get("span") or "").strip()

    # ── type: normalise to AssumptionType
    raw_type = str(item.get("type") or "domain").strip().lower()
    a_type = _TYPE_MAP.get(raw_type, AssumptionType.DOMAIN)

    # ── explicit: "EXPLICIT" → True, anything else → False
    raw_explicit = item.get("explicitness") or item.get("explicit")
    if isinstance(raw_explicit, bool):
        explicit = raw_explicit
    else:
        explicit = str(raw_explicit).strip().upper() == "EXPLICIT"

    return Assumption(
        type=a_type,
        constraint=constraint,
        explicit=explicit,
        span=span,
        verification=VerificationStatus.WEAK,
        score=0.0,
    )


def extract_assumptions(
    text: str,
    claims: List[Claim],
    retriever=None,          # DocumentRetriever | None
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
        max_tokens=600,
    )

    assumptions: List[Assumption] = []
    if not result:
        logger.warning("Agent 6: No result from LLM.")
        return assumptions

    # Accept {"assumptions": [...]} or a bare list
    raw_list = result.get("assumptions", result) if isinstance(result, dict) else result
    if not isinstance(raw_list, list):
        logger.warning("Agent 6: Unexpected response shape: %s", type(raw_list))
        return assumptions

    for item in raw_list:
        a = _parse_assumption(item)
        if a is not None:
            assumptions.append(a)

    logger.info("Agent 6: Extracted %d assumptions.", len(assumptions))
    return assumptions


def assign_assumptions_to_claims(claims: List[Claim], assumptions: List[Assumption]) -> List[Claim]:
    """Keyword-overlap matching — zero LLM tokens."""
    import re
    for claim in claims:
        claim_words = set(re.split(r'\W+', f"{claim.subject} {claim.object} {claim.method}".lower()))
        for assumption in assumptions:
            constraint_words = set(
                w for w in re.split(r'\W+', assumption.constraint.lower()) if len(w) > 3
            )
            if len(constraint_words & claim_words) >= 1:
                claim.assumptions.append(assumption)
    return claims
