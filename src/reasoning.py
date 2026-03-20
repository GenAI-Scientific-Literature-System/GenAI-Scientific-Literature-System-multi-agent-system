"""
MERLIN Executable Assumption Reasoning
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Deterministic, zero-token reasoning on MERLINStruct — never on raw text.

Core: set-operation assumption agreement
    if  A1 == A2             → "support"       (identical assumption sets)
    elif A1.isdisjoint(A2)   → "contradiction"  (no shared assumptions)
    else                     → "conditional"    (partial overlap)

This is the explicit, publishable version of R(Ci, Cj | Ai, Aj).
"""
import re
import logging
from typing import Tuple, List, Set

from src.models.schemas import Assumption

logger = logging.getLogger(__name__)


# ── Antonym table for conflict detection ─────────────────────────────────────
_ANTONYMS: List[Tuple[Set[str], Set[str]]] = [
    ({"high", "resource"},      {"low", "resource"}),
    ({"large", "big"},          {"small", "tiny"}),
    ({"online", "realtime"},    {"offline", "batch"}),
    ({"supervised"},            {"unsupervised", "selfsupervised"}),
    ({"gpu", "cuda"},           {"cpu"}),
    ({"clean", "curated"},      {"noisy", "raw"}),
    ({"english", "monolingual"},{"multilingual", "crosslingual"}),
    ({"centralised"},           {"distributed", "federated"}),
    ({"balanced"},              {"imbalanced", "skewed"}),
]


def _tok(text: str) -> Set[str]:
    return {t for t in re.split(r"\W+", text.lower()) if len(t) >= 3}


def _antonym_conflict(t1: Set[str], t2: Set[str]) -> bool:
    for pos, neg in _ANTONYMS:
        if (pos & t1 and neg & t2) or (neg & t1 and pos & t2):
            return True
    return False


# ── Core set-operation reasoning function ─────────────────────────────────────

def assumption_agreement(a1_set: Set[str], a2_set: Set[str]) -> Tuple[str, float]:
    """
    Explicit set-operation agreement function — the formal R(Ci, Cj | Ai, Aj).

    if   A1 == A2              → "support"       (conf = 1.0)
    elif A1.isdisjoint(A2)     → "contradiction"  (conf = 0.9 if antonym else 0.65)
    else                       → "conditional"    (conf = Jaccard overlap)
    """
    if not a1_set or not a2_set:
        return "support", 0.5

    if a1_set == a2_set:
        return "support", 1.0

    if a1_set.isdisjoint(a2_set):
        conf = 0.9 if _antonym_conflict(a1_set, a2_set) else 0.65
        return "contradiction", conf

    jaccard = len(a1_set & a2_set) / len(a1_set | a2_set)
    return "conditional", round(jaccard, 3)


def struct_agreement(struct, ci_id: str, cj_id: str) -> Tuple[str, float]:
    """Pull assumption sets from MERLINStruct, run set-op reasoning. No text."""
    a1 = struct.assumption_set(ci_id)
    a2 = struct.assumption_set(cj_id)
    return assumption_agreement(a1, a2)


# ── Legacy list-based interface (used by agent6 assign step) ──────────────────

def assumption_relation(
    a1_list: List[Assumption],
    a2_list: List[Assumption],
) -> Tuple[str, float]:
    a1_set, a2_set = set(), set()
    for a in a1_list:
        a1_set |= _tok(a.constraint)
    for a in a2_list:
        a2_set |= _tok(a.constraint)

    rel, conf = assumption_agreement(a1_set, a2_set)
    rel_map = {"support": "compatible", "contradiction": "conflict", "conditional": "conditional"}
    return rel_map.get(rel, rel), conf


# ── Formal epistemic loss ─────────────────────────────────────────────────────

def formal_score(
    contradiction_count: int,
    total_pairs: int,
    avg_uncertainty: float,
    assumption_rejection_rate: float,
) -> float:
    cp = (contradiction_count / max(total_pairs, 1)) * 0.40
    up = avg_uncertainty * 0.40
    ap = assumption_rejection_rate * 0.20
    return round(min(cp + up + ap, 1.0), 3)
