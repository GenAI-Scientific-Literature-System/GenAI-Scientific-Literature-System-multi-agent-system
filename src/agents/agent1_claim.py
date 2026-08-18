"""
Agent 1: Claim Extraction  C = (S, P, O, M, D, Θ)
[V1 Anti-hallucination] grounding check after extraction.

UPGRADE: accepts a DocumentRetriever — sends only the relevant chunks
         (~200 chars) instead of the full sanitised text (~1800 chars).
         Token reduction: ~80 %.
"""
import logging
import re
from typing import List, Tuple, Optional
from src.models.schemas import Claim
from src.llm_client import call_llm, sanitize_for_prompt
from src.hallucination_guard import filter_hallucinated_claims
from config import CLAIM_PROMPT

logger = logging.getLogger(__name__)

# Query used to retrieve claim-relevant chunks
_CLAIM_QUERY = "main claims findings results contributions conclusions"


def _heuristic_claims(text: str, paper_id: str) -> List[Claim]:
    """Grounded fallback for when the configured clinical LLM is unavailable.

    It intentionally only emits sentences containing a result-direction cue;
    every field is derived from that sentence, so the normal V1 guard can
    reject it just like an LLM-produced claim.
    """
    cues = re.compile(r"\b(reduce[sd]?|decrease[sd]?|improve[sd]?|increase[sd]?|associated with|no (?:significant )?(?:effect|difference)|effective|ineffective)\b", re.I)
    split = re.compile(r"(?<=[.!?])\s+")
    claims: List[Claim] = []
    for sentence in split.split(text or ""):
        match = cues.search(sentence)
        if not match:
            continue
        words_before = sentence[:match.start()].strip().split()
        if not words_before:
            continue
        # Take the most relevant subject phrase directly preceding the predicate
        subject = " ".join(words_before[-4:])
        predicate = match.group(0).lower()
        object_ = sentence[match.end():].strip(" .;:")[:180]
        if subject and object_:
            claims.append(Claim(
                subject=subject, predicate=predicate, object=object_,
                domain="clinical", paper_id=paper_id, extraction_confidence=0.85,
            ))
        if len(claims) >= 5:
            break
    return claims


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

    result = call_llm(
        prompt,
        system="You extract scientific claims. Return only a JSON object with a 'claims' array. No prose.",
        max_tokens=700,
    )

    raw_claims: List[Claim] = []
    if not result:
        # 1. Primary path: BioBERT Transformer NER Extraction
        try:
            from services.common.biobert import BioBERTClaimExtractor
            extractor = BioBERTClaimExtractor(auto_load=True)
            biobert_raw = extractor.extract(text, paper_id=paper_id)
            if biobert_raw and extractor._is_loaded:
                for b_item in biobert_raw:
                    raw_claims.append(Claim(
                        id=b_item.get("id", ""),
                        subject=b_item.get("subject", ""),
                        predicate=b_item.get("predicate", ""),
                        object=b_item.get("object", ""),
                        domain="clinical",
                        paper_id=paper_id,
                        extraction_confidence=b_item.get("extraction_confidence", 0.85),
                    ))
                logger.info("Agent 1: Extracted %d claims via BioBERT NER (paper='%s').", len(raw_claims), paper_id)
        except Exception as e:
            logger.debug("BioBERT inference exception in Agent 1: %s", e)

        # 2. Fallback to Grounded Heuristic if BioBERT is not available
        if not raw_claims:
            logger.warning("Agent 1: LLM/BioBERT unavailable; using grounded clinical heuristic (paper='%s').", paper_id)
            raw_claims = _heuristic_claims(text, paper_id)

        grounded, dropped, _ = filter_hallucinated_claims(raw_claims, text)
        return grounded, dropped

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
            extraction_confidence=float(item.get("confidence", 0.5) or 0.5),
        )
        raw_claims.append(c)

    # [V1] Grounding check against original (unsanitized) text
    grounded, dropped, _ = filter_hallucinated_claims(raw_claims, text)
    logger.info(
        "Agent 1: %d claims extracted, %d dropped by V1 (paper='%s').",
        len(grounded), dropped, paper_id,
    )
    return grounded, dropped
