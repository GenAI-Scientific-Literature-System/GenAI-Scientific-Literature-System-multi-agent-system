"""
Agent 6.1: Assumption Verification — Anti-Hallucination Core
[V5 Anti-hallucination]
Tier 1: exact span match         → VERIFIED (1.0)
Tier 2: bigram overlap >= 0.50   → VERIFIED (score)
Tier 3: unigram overlap >= 0.40  → WEAK     (score)
Tier 4: local NLI (implicit only)→ upgrade/downgrade
Else:                            → REJECTED (filtered out)
Zero Mistral tokens.
"""
import logging
from typing import List
from src.models.schemas import Assumption, VerificationStatus
from src.hallucination_guard import deep_assumption_ground
from config import NLI_THRESHOLD

logger = logging.getLogger(__name__)

_pipeline = None


def _get_nli_pipeline():
    global _pipeline
    if _pipeline is None:
        try:
            from transformers import pipeline
            from config import NLI_MODEL_PRIMARY
            _pipeline = pipeline("zero-shot-classification", model=NLI_MODEL_PRIMARY, device=-1)
            logger.info("Agent 6.1: NLI pipeline loaded.")
        except Exception as e:
            logger.warning("Agent 6.1: NLI unavailable (%s). Guard-only mode.", e)
            _pipeline = "unavailable"
    return _pipeline


def verify_assumption(assumption: Assumption, source_text: str) -> Assumption:
    """[V5] Four-tier verification. Tiers 1-3 are free; Tier 4 is local NLI."""
    status, score = deep_assumption_ground(assumption, source_text)

    if status == VerificationStatus.VERIFIED:
        assumption.verification = status
        assumption.score = score
        return assumption

    pipe = _get_nli_pipeline()
    if pipe and pipe != "unavailable" and not assumption.explicit:
        try:
            res = pipe(source_text[:512],
                       candidate_labels=[assumption.constraint],
                       hypothesis_template="This text assumes {}.")
            nli_score = res["scores"][0] if res["scores"] else 0.0
            if nli_score >= 0.85:
                status = VerificationStatus.VERIFIED
                score  = round(nli_score, 3)
            elif nli_score >= NLI_THRESHOLD and status != VerificationStatus.REJECTED:
                status = VerificationStatus.WEAK
                score  = max(score, round(nli_score, 3))
            elif status == VerificationStatus.WEAK and nli_score < NLI_THRESHOLD * 0.7:
                status = VerificationStatus.REJECTED
                score  = round(nli_score, 3)
        except Exception as e:
            logger.debug("Agent 6.1: NLI error: %s", e)

    assumption.verification = status
    assumption.score = score
    return assumption


def verify_all_assumptions(assumptions: List[Assumption], source_text: str) -> List[Assumption]:
    """Run V5 guard on all assumptions; filter REJECTED."""
    verified_c, weak_c, rejected_c = 0, 0, 0
    kept: List[Assumption] = []

    for a in assumptions:
        a = verify_assumption(a, source_text)
        if a.verification == VerificationStatus.VERIFIED:
            verified_c += 1; kept.append(a)
        elif a.verification == VerificationStatus.WEAK:
            weak_c += 1; kept.append(a)
        else:
            rejected_c += 1
            logger.debug("[V5] Rejected: '%s' (score=%.2f)", a.constraint[:60], a.score)

    logger.info("Agent 6.1 [V5]: verified=%d weak=%d rejected=%d",
                verified_c, weak_c, rejected_c)
    return kept
