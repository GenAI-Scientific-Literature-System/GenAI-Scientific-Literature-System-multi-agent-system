"""
Agent 2: Evidence Attribution
[V2 Anti-hallucination] span verification after attribution.

UPGRADE: retriever fetches the most relevant chunk per claim
         instead of sending the full document for every call.
"""
import logging
from typing import List, Tuple, Optional
from src.models.schemas import Claim
from src.mistral_client import call_mistral, sanitize_for_prompt
from src.hallucination_guard import verify_evidence_spans
from config import EVIDENCE_PROMPT

logger = logging.getLogger(__name__)


def _compact_claims(claims: List[Claim]) -> str:
    parts = [f'{i}:"{c.subject} {c.predicate} {c.object}"' for i, c in enumerate(claims)]
    return "[" + ",".join(parts) + "]"


def attribute_evidence(
    claims: List[Claim],
    text: str,
    retriever=None,               # DocumentRetriever | None
) -> Tuple[List[Claim], int]:

    if not claims:
        return claims, 0

    compact = _compact_claims(claims)

    # ── Context: retrieved (per-claim query) or full text ────────────────────
    if retriever is not None:
        # Build a single combined query from all claim texts
        claim_query = " ".join(f"{c.subject} {c.object}" for c in claims[:4])
        context = retriever.retrieve(claim_query, top_k=3)
        context = sanitize_for_prompt(context, max_chars=900)
    else:
        context = sanitize_for_prompt(text, max_chars=1400)

    prompt = EVIDENCE_PROMPT.format(claims=compact, text=context)

    result = call_mistral(
        prompt,
        system="Return only a JSON object with an 'evidence' array. No prose.",
        max_tokens=600,
    )

    if not result:
        logger.warning("Agent 2: No result, skipping evidence attribution.")
        return claims, 0

    raw_list = result.get("evidence", result) if isinstance(result, dict) else result
    if not isinstance(raw_list, list):
        raw_list = []

    for item in raw_list:
        idx = item.get("claim_id")
        if idx is None or not isinstance(idx, int) or idx >= len(claims):
            continue
        claims[idx].evidence_spans    = item.get("spans", [])
        claims[idx].evidence_strength = item.get("strength", "medium")

    # [V2] Verify spans exist in original text
    removed = 0
    for c in claims:
        before = len(c.evidence_spans)
        c      = verify_evidence_spans(c, text)
        removed += before - len(c.evidence_spans)

    if removed:
        logger.info("Agent 2: V2 removed %d fabricated spans.", removed)
    return claims, removed
