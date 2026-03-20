"""
Agent 1: Claim Extraction  C = (S, P, O, M, D, Θ)
[V1 Anti-hallucination] grounding check after extraction.

UPGRADE: accepts a DocumentRetriever — sends only the relevant chunks
         (~200 chars) instead of the full sanitised text (~1800 chars).
         Token reduction: ~80 %.
"""
import logging
from typing import List, Tuple, Optional
from src.models.schemas import Claim
from src.mistral_client import call_mistral, sanitize_for_prompt
from src.hallucination_guard import filter_hallucinated_claims
from config import CLAIM_PROMPT

logger = logging.getLogger(__name__)

# Query used to retrieve claim-relevant chunks
_CLAIM_QUERY = "main claims findings results contributions conclusions"


def extract_claims(
    text: str,
    paper_id: str = "",
    retriever=None,               # DocumentRetriever | None
) -> Tuple[List[Claim], int]:

    # ── Context: retrieved chunks (RAG) or sanitised full text ───────────────
    if retriever is not None:
        context = retriever.retrieve(_CLAIM_QUERY, top_k=4)
        context = sanitize_for_prompt(context, max_chars=900)
        logger.debug("Agent 1 [RAG]: using %d chars of retrieved context.", len(context))
    else:
        context = sanitize_for_prompt(text, max_chars=1800)

    prompt = CLAIM_PROMPT.format(text=context)

    result = call_mistral(
        prompt,
        system="You extract scientific claims. Return only a JSON object with a 'claims' array. No prose.",
        max_tokens=700,
    )

    raw_claims: List[Claim] = []
    if not result:
        logger.warning("Agent 1: No result from Mistral (paper='%s').", paper_id)
        return raw_claims, 0

    raw_list = result.get("claims", result) if isinstance(result, dict) else result
    if not isinstance(raw_list, list):
        raw_list = []

    for item in raw_list:
        if not isinstance(item, dict):
            continue
        subj = str(item.get("subject", "")).strip()
        pred = str(item.get("predicate", "")).strip()
        if not subj or not pred:
            continue
        c = Claim(
            subject=subj,
            predicate=pred,
            object=str(item.get("object", "")).strip(),
            method=str(item.get("method", "")).strip(),
            domain=str(item.get("domain", "")).strip(),
            paper_id=paper_id,
        )
        raw_claims.append(c)

    # [V1] Grounding check against original (unsanitized) text
    grounded, dropped, _ = filter_hallucinated_claims(raw_claims, text)
    logger.info(
        "Agent 1: %d claims extracted, %d dropped by V1 (paper='%s').",
        len(grounded), dropped, paper_id,
    )
    return grounded, dropped
