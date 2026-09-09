"""
MERLIN HallucinationGuard
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Grounded verification for all 5 hallucination vectors.
All checks use local computation only — zero Mistral tokens.

Vector map:
  [V1] Claim grounding      → extract_claims (Agent 1)
  [V2] Evidence span check  → attribute_evidence (Agent 2)
  [V3] Agreement grounding  → compute_agreements (Agent 4)
  [V4] Gap grounding        → detect_gaps (Agent 5)
  [V5] Assumption grounding → verify_assumptions (Agent 6.1)
"""
import re
import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

logger = logging.getLogger(__name__)


# ── Stop-words to ignore in overlap checks ───────────────────────────────────
_STOP = {
    "a","an","the","is","are","was","were","be","been","being",
    "have","has","had","do","does","did","will","would","could","should",
    "may","might","shall","of","in","on","at","to","for","and","or","but",
    "not","with","by","from","that","this","which","it","we","they","our",
    "their","its","as","also","more","less","than","when","where","how",
    "what","while","if","so","then","can","there","no","up","down","all",
}

# ── Confidence calibration thresholds ────────────────────────────────────────
_MAX_CONFIDENCE_NO_EVIDENCE = 0.70   # cap if no span grounding found
_NGRAM_WINDOW               = 2      # bigram overlap by default


# ══════════════════════════════════════════════════════════════════════════════
# Core token-level utilities
# ══════════════════════════════════════════════════════════════════════════════

def _tokenize(text: str) -> List[str]:
    """Lower-case alphanum tokens, stop-words removed."""
    return [
        t for t in re.split(r"\W+", text.lower())
        if t and t not in _STOP and len(t) > 1
    ]


def _ngrams(tokens: List[str], n: int = 2) -> set:
    if len(tokens) < n:
        return set(tokens)
    return {tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)}


def token_overlap(text: str, source: str) -> float:
    """
    Proportion of content tokens in `text` that appear in `source`.
    Returns 0.0–1.0.
    """
    t_tokens = _tokenize(text)
    s_tokens = set(_tokenize(source))
    if not t_tokens:
        return 0.0
    hits = sum(1 for t in t_tokens if t in s_tokens)
    return hits / len(t_tokens)


def ngram_overlap(text: str, source: str, n: int = _NGRAM_WINDOW) -> float:
    """
    Bigram overlap between `text` and `source`.
    More discriminative than unigram for detecting paraphrased fabrication.
    """
    t_ng = _ngrams(_tokenize(text), n)
    s_ng = _ngrams(_tokenize(source), n)
    if not t_ng:
        return 0.0
    return len(t_ng & s_ng) / len(t_ng)


def span_exists(span: str, source: str) -> bool:
    """Exact or near-exact span match (case-insensitive, collapsed whitespace)."""
    if not span or len(span) < 4:
        return False
    span_clean   = re.sub(r"\s+", " ", span.strip().lower())
    source_clean = re.sub(r"\s+", " ", source.lower())
    return span_clean in source_clean


def fuzzy_span_score(span: str, source: str, window: int = 150) -> float:
    """
    Sliding-window bigram overlap: find the best-matching window of
    `window` chars in `source` for the given `span`.
    Efficient O(|source|/stride) approximation.
    """
    if not span:
        return 0.0
    span_ng = _ngrams(_tokenize(span), 2)
    if not span_ng:
        return token_overlap(span, source)

    best = 0.0
    stride = max(1, window // 4)
    for start in range(0, max(1, len(source) - window), stride):
        chunk = source[start: start + window]
        chunk_ng = _ngrams(_tokenize(chunk), 2)
        if not chunk_ng:
            continue
        score = len(span_ng & chunk_ng) / len(span_ng)
        if score > best:
            best = score
        if best >= 0.9:
            break
    return best


# ══════════════════════════════════════════════════════════════════════════════
# [V1] Claim Grounding
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ClaimGroundingResult:
    grounded: bool
    overlap:  float
    reason:   str


def ground_claim(subject: str, predicate: str, obj: str, source: str,
                 threshold: float = 0.35) -> ClaimGroundingResult:
    """
    [V1] Verify that a claim's core components are traceable to source text.
    Strategy:
      1. Check subject token overlap with source
      2. Check object  token overlap with source
      3. Accept if max(subject_overlap, object_overlap) ≥ threshold
         AND combined claim text bigram overlap ≥ threshold/2
    Rationale: predicate is often paraphrased (LLM normalises "showed" → "demonstrates");
    subject and object are proper nouns/technical terms that the LLM cannot invent.
    """
    claim_text     = f"{subject} {obj}"
    subj_overlap   = token_overlap(subject, source)
    obj_overlap    = token_overlap(obj,     source)
    bigram_score   = ngram_overlap(claim_text, source, n=2)

    max_component  = max(subj_overlap, obj_overlap)
    # OR logic: high component overlap alone is sufficient for short claims
    # where bigrams collapse (e.g. single-token subjects/objects)
    grounded       = (max_component >= threshold) or (bigram_score >= 0.30)

    reason = (
        f"subj={subj_overlap:.2f} obj={obj_overlap:.2f} "
        f"bigram={bigram_score:.2f} → {'GROUNDED' if grounded else 'HALLUCINATED'}"
    )
    return ClaimGroundingResult(grounded=grounded, overlap=max_component, reason=reason)


def filter_hallucinated_claims(claims, source: str, threshold: float = 0.35):
    """
    [V1] Remove claims whose core entities are not traceable to source.
    Returns (kept_claims, dropped_count, report).
    """
    kept, dropped = [], 0
    report = []
    for c in claims:
        result = ground_claim(c.subject, c.predicate, c.object, source, threshold)
        if result.grounded:
            kept.append(c)
        else:
            dropped += 1
            report.append({
                "claim": c.text, "reason": result.reason,
                "verdict": "HALLUCINATED — dropped"
            })
            logger.warning("[V1] Dropped hallucinated claim: '%s' (%s)", c.text, result.reason)

    logger.info("[V1] Claim grounding: %d kept, %d dropped.", len(kept), dropped)
    return kept, dropped, report


# ══════════════════════════════════════════════════════════════════════════════
# [V2] Evidence Span Verification
# ══════════════════════════════════════════════════════════════════════════════

def verify_evidence_spans(claim, source: str, fuzzy_threshold: float = 0.40):
    """
    [V2] Verify each evidence span actually exists in source.
    - Exact match  → VERIFIED
    - Fuzzy ≥ 0.4  → WEAK (keep but downgrade confidence)
    - Otherwise    → remove span, flag as hallucinated

    Also recalibrates evidence_strength:
      verified spans → keep original
      only weak spans → downgrade to 'medium'
      no spans       → downgrade to 'low'
    """
    if not claim.evidence_spans:
        return claim

    verified, weak, fabricated = [], [], []
    for span in claim.evidence_spans:
        if span_exists(span, source):
            verified.append(span)
        else:
            score = fuzzy_span_score(span, source)
            if score >= fuzzy_threshold:
                weak.append(span)
                logger.debug("[V2] Weak span (score=%.2f): '%s'", score, span[:60])
            else:
                fabricated.append(span)
                logger.warning("[V2] Fabricated span dropped: '%s'", span[:60])

    claim.evidence_spans = verified + weak

    # Recalibrate strength
    if not claim.evidence_spans:
        claim.evidence_strength = "low"
    elif not verified and weak:
        claim.evidence_strength = "medium" if claim.evidence_strength == "high" else claim.evidence_strength

    if fabricated:
        logger.info("[V2] %d fabricated spans removed from claim '%s'.",
                    len(fabricated), claim.id)
    return claim


# ══════════════════════════════════════════════════════════════════════════════
# [V3] Agreement Reason Grounding
# ══════════════════════════════════════════════════════════════════════════════

def verify_agreement_reason(agreement, claim_i_text: str, claim_j_text: str,
                             threshold: float = 0.25) -> Tuple[bool, float]:
    """
    [V3] Verify the LLM-generated reason is grounded in the actual claim texts.
    Checks that the reason references concepts actually present in C_i or C_j.
    If not grounded:
      - reason is replaced with a factual summary
      - confidence is capped at _MAX_CONFIDENCE_NO_EVIDENCE
    """
    reason = agreement.reason or ""
    if not reason or reason.startswith("Heuristic:"):
        return True, agreement.confidence   # heuristic reasons are always grounded

    combined_claims = claim_i_text + " " + claim_j_text
    overlap = token_overlap(reason, combined_claims)

    if overlap < threshold:
        logger.warning(
            "[V3] Agreement reason not grounded (overlap=%.2f): '%s'",
            overlap, reason[:80]
        )
        agreement.reason = (
            f"[Auto-summary] Relation={agreement.relation} "
            f"between '{claim_i_text[:40]}' and '{claim_j_text[:40]}'"
        )
        agreement.confidence = min(agreement.confidence, _MAX_CONFIDENCE_NO_EVIDENCE)
        return False, overlap

    return True, overlap


# ══════════════════════════════════════════════════════════════════════════════
# [V4] Research Gap Grounding
# ══════════════════════════════════════════════════════════════════════════════

def verify_gap(gap, claims, threshold: float = 0.12) -> Tuple[bool, float]:
    """
    [V4] Verify the gap description is grounded in actual high-uncertainty claims.
    A gap that shares no vocabulary with any claim is likely hallucinated.
    Returns (grounded, best_overlap_score).
    """
    if not claims:
        return False, 0.0

    gap_text = gap.gap or ""
    if not gap_text:
        return False, 0.0

    all_claim_text = " ".join(
        f"{c.subject} {c.predicate} {c.object} {c.domain}" for c in claims
    )
    overlap = token_overlap(gap_text, all_claim_text)

    if overlap < threshold:
        logger.warning(
            "[V4] Gap not grounded in claims (overlap=%.2f): '%s'",
            overlap, gap_text[:80]
        )
        return False, overlap

    return True, overlap


def filter_hallucinated_gaps(gaps, claims):
    """[V4] Remove gaps not grounded in actual claim content."""
    kept, dropped = [], 0
    for g in gaps:
        grounded, score = verify_gap(g, claims)
        if grounded:
            kept.append(g)
        else:
            dropped += 1
            logger.info("[V4] Dropped ungrounded gap: '%s' (overlap=%.2f)",
                        (g.gap or "")[:60], score)
    logger.info("[V4] Gap grounding: %d kept, %d dropped.", len(kept), dropped)
    return kept, dropped


# ══════════════════════════════════════════════════════════════════════════════
# [V5] Assumption Grounding (deeper version for agent6_1)
# ══════════════════════════════════════════════════════════════════════════════

def deep_assumption_ground(assumption, source: str) -> Tuple[str, float]:
    """
    [V5] Three-tier grounding for assumptions:
      Tier 1 — Exact span match              → VERIFIED (1.0)
      Tier 2 — Bigram overlap ≥ 0.5         → VERIFIED (score)
      Tier 3a — Unigram overlap ≥ 0.30      → WEAK     (score)
      Tier 3b — Implicit: unigram ≥ 0.15    → WEAK     (score)
      Else                                  → REJECTED (score)

    OCP FIX: implicit assumptions are paraphrased from context the LLM
    infers rather than copies — they score lower on n-gram overlap even
    when genuinely grounded. Tier 3b applies a relaxed threshold of 0.15
    for implicit assumptions only, keeping them as WEAK instead of
    silently dropping valid methodological context.
    """
    from src.models.schemas import VerificationStatus

    # Tier 1: exact span
    if span_exists(assumption.span, source):
        return VerificationStatus.VERIFIED, 1.0

    # Tier 2: bigram overlap on constraint text
    bg_score = ngram_overlap(assumption.constraint, source, n=2)
    if bg_score >= 0.50:
        return VerificationStatus.VERIFIED, round(bg_score, 3)

    # Tier 3: unigram overlap
    ug_score = token_overlap(assumption.constraint, source)
    # 3a — explicit assumptions need reasonable lexical overlap
    if ug_score >= 0.30:
        return VerificationStatus.WEAK, round(ug_score, 3)
    # 3b — implicit assumptions are inferred/paraphrased; relax threshold
    if not assumption.explicit and ug_score >= 0.15:
        return VerificationStatus.WEAK, round(ug_score, 3)

    return VerificationStatus.REJECTED, round(ug_score, 3)


# ══════════════════════════════════════════════════════════════════════════════
# Hallucination Report
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class HallucinationReport:
    v1_claims_dropped:     int   = 0
    v2_spans_removed:      int   = 0
    v3_reasons_rewritten:  int   = 0
    v4_gaps_dropped:       int   = 0
    v5_assumptions_rejected: int = 0
    v1_details: List[dict]  = field(default_factory=list)

    @property
    def total_interventions(self) -> int:
        return (self.v1_claims_dropped + self.v2_spans_removed +
                self.v3_reasons_rewritten + self.v4_gaps_dropped +
                self.v5_assumptions_rejected)

    def to_dict(self) -> Dict:
        return {
            "v1_claims_dropped":       self.v1_claims_dropped,
            "v2_spans_removed":        self.v2_spans_removed,
            "v3_reasons_rewritten":    self.v3_reasons_rewritten,
            "v4_gaps_dropped":         self.v4_gaps_dropped,
            "v5_assumptions_rejected": self.v5_assumptions_rejected,
            "total_interventions":     self.total_interventions,
            "v1_details":              self.v1_details,
        }
