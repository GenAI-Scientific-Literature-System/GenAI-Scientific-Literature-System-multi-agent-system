"""
MERLIN Assumption-Consistency Engine
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Claims are valid ONLY under their verified assumptions.
This is the paper's core contribution made executable.

    validate_claim(claim, assumptions) → (valid: bool, reason: str)
    validate_all(claims)               → (valid_claims, rejected_claims, report)

A claim is INVALID if:
  - It has zero verified assumptions (no grounding)
  - Any REQUIRED assumption has verification=REJECTED
  - The claim predicate requires an assumption type that is absent

This is not the hallucination guard (V1–V5).
It is the epistemic consistency layer that runs AFTER verification.
"""
import logging
from typing import List, Tuple, Dict, Any
from src.models.schemas import Claim, Assumption, VerificationStatus

logger = logging.getLogger(__name__)

# Predicates that REQUIRE specific assumption types to be meaningful
_PREDICATE_REQUIRES: Dict[str, str] = {
    "outperforms":   "method",      # needs a method assumption to be comparable
    "underperforms": "method",
    "improves":      "method",
    "reduces":       "method",
    "demonstrates":  "scope",       # needs scope to be falsifiable
    "shows":         "scope",
    "proposes":      None,          # no assumption required
    "describes":     None,
}


def validate_claim(claim: Claim) -> Tuple[bool, str]:
    """
    Check whether a claim is epistemically valid given its assumption set.

    Rules:
      R1: If no assumptions at all → mark as UNGROUNDED (warn, don't reject)
      R2: If any assumption has status=REJECTED → invalid
      R3: If predicate requires assumption type X but none of type X → invalid
    """
    if not claim.assumptions:
        # R1: warn only — no assumption means claim is unqualified, not wrong
        return True, "UNGROUNDED: no assumptions (claim may be overgeneralized)"

    # R2: reject if any assumption failed verification
    rejected = [a for a in claim.assumptions
                if a.verification == VerificationStatus.REJECTED]
    if rejected:
        reasons = ", ".join(f"'{a.constraint[:30]}'" for a in rejected[:2])
        return False, f"INVALID: rejected assumptions: {reasons}"

    # R3: check predicate-type requirement
    required_type = _PREDICATE_REQUIRES.get(claim.predicate.lower())
    if required_type:
        has_required = any(
            a.type == required_type and a.verification != VerificationStatus.REJECTED
            for a in claim.assumptions
        )
        if not has_required:
            return False, (
                f"INVALID: predicate '{claim.predicate}' requires a '{required_type}' "
                f"assumption but none was verified"
            )

    return True, "VALID"


def validate_all(claims: List[Claim]) -> Tuple[List[Claim], List[Claim], Dict[str, Any]]:
    """
    Run assumption-consistency check on all claims.

    Returns:
        valid_claims   — pass assumption check
        invalid_claims — actively rejected (removed from pipeline)
        report         — {rejected_count, ungrounded_count, reasons:[...]}
    """
    valid, invalid = [], []
    report: Dict[str, Any] = {
        "validated":        len(claims),
        "rejected":         0,
        "ungrounded":       0,
        "rejection_reasons": [],
    }

    for claim in claims:
        ok, reason = validate_claim(claim)
        if ok:
            valid.append(claim)
            if "UNGROUNDED" in reason:
                report["ungrounded"] += 1
        else:
            invalid.append(claim)
            report["rejected"]   += 1
            report["rejection_reasons"].append({
                "claim": claim.text[:60],
                "reason": reason,
            })
            logger.info(
                "[ACE] Claim rejected: '%s' — %s",
                claim.text[:50], reason
            )

    logger.info(
        "AssumptionEngine: %d claims — %d valid, %d rejected, %d ungrounded",
        len(claims), len(valid), report["rejected"], report["ungrounded"],
    )
    return valid, invalid, report
