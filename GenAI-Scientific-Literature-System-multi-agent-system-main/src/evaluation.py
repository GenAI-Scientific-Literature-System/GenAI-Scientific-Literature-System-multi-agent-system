"""
MERLIN Evaluation
━━━━━━━━━━━━━━━━
Computable metrics: precision, recall, F1 for claims and gaps.
Can be run against the bench sample in data/merlin_bench_sample.json.

Usage:
    from src.evaluation import evaluate_claims, evaluate_gaps, run_bench

    scores = run_bench(pipeline_result, ground_truth)
    # → {"claim_f1": 0.74, "gap_recall": 0.80, ...}
"""
import json
import re
import logging
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)


# ── Token helpers ─────────────────────────────────────────────────────────────

def _tok(text: str) -> set:
    return {t.lower() for t in re.split(r"\W+", text) if len(t) > 2}


def _overlap_score(pred: str, truth: str) -> float:
    """Token-level overlap (soft match)."""
    pt, tt = _tok(pred), _tok(truth)
    if not pt or not tt:
        return 0.0
    return len(pt & tt) / len(pt | tt)


def _match(pred: str, truth_list: List[str], threshold: float = 0.35) -> bool:
    return any(_overlap_score(pred, t) >= threshold for t in truth_list)


# ── Core metrics ──────────────────────────────────────────────────────────────

def precision_recall_f1(
    predicted: List[str],
    ground_truth: List[str],
    threshold: float = 0.35,
) -> Dict[str, float]:
    """
    Soft-match P/R/F1 between predicted and ground-truth string lists.
    A prediction counts as TP if it matches any ground-truth item
    with token overlap >= threshold.
    """
    if not predicted and not ground_truth:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not predicted:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    if not ground_truth:
        return {"precision": 0.0, "recall": 1.0, "f1": 0.0}

    tp_pred = sum(1 for p in predicted if _match(p, ground_truth, threshold))
    tp_truth = sum(1 for t in ground_truth if _match(t, predicted, threshold))

    precision = tp_pred / len(predicted)
    recall    = tp_truth / len(ground_truth)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 3),
        "recall":    round(recall, 3),
        "f1":        round(f1, 3),
    }


# ── Domain-specific evaluators ────────────────────────────────────────────────

def evaluate_claims(
    predicted_claims,       # List[Claim]
    ground_truth_texts: List[str],
) -> Dict[str, float]:
    pred_texts = [f"{c.subject} {c.predicate} {c.object}" for c in predicted_claims]
    scores = precision_recall_f1(pred_texts, ground_truth_texts)
    logger.info("Claim eval: P=%.2f R=%.2f F1=%.2f", scores["precision"], scores["recall"], scores["f1"])
    return scores


def evaluate_gaps(
    predicted_gaps,         # List[ResearchGap]
    ground_truth_gaps: List[str],
) -> Dict[str, float]:
    pred_texts = [g.gap for g in predicted_gaps]
    scores = precision_recall_f1(pred_texts, ground_truth_gaps)
    logger.info("Gap eval:   P=%.2f R=%.2f F1=%.2f", scores["precision"], scores["recall"], scores["f1"])
    return scores


def evaluate_relations(
    predicted_agreements,   # List[Agreement]
    ground_truth: List[Dict],  # [{"i": id, "j": id, "relation": str}]
) -> Dict[str, float]:
    """
    Exact relation-type match for (Ci, Cj) pairs that appear in ground truth.
    """
    if not ground_truth:
        return {"accuracy": 0.0}

    ag_map = {(a.claim_i_id, a.claim_j_id): a.relation for a in predicted_agreements}
    ag_map.update({(a.claim_j_id, a.claim_i_id): a.relation for a in predicted_agreements})

    correct = 0
    for gt in ground_truth:
        key = (gt.get("i", ""), gt.get("j", ""))
        pred_rel = ag_map.get(key, ag_map.get((key[1], key[0]), None))
        if pred_rel and pred_rel == gt.get("relation", ""):
            correct += 1

    acc = correct / len(ground_truth)
    logger.info("Relation eval: accuracy=%.2f (%d/%d)", acc, correct, len(ground_truth))
    return {"accuracy": round(acc, 3)}


# ── Bench runner ──────────────────────────────────────────────────────────────

def run_bench(result_dict: Dict[str, Any], bench_path: str = "data/merlin_bench_sample.json") -> Dict[str, Any]:
    """
    Run evaluation against the bench sample.
    Returns full score report.
    """
    try:
        with open(bench_path) as f:
            bench = json.load(f)
    except Exception as e:
        logger.warning("Bench load failed: %s", e)
        return {"error": str(e)}

    report: Dict[str, Any] = {}

    # Claim eval
    if "ground_claims" in bench:
        pred_claims = result_dict.get("claims", [])
        pred_texts  = [f"{c.get('subject','')} {c.get('predicate','')} {c.get('object','')}"
                       for c in pred_claims]
        report["claims"] = precision_recall_f1(pred_texts, bench["ground_claims"])

    # Gap eval
    if "ground_gaps" in bench:
        pred_gaps = [g.get("gap", "") for g in result_dict.get("gaps", [])]
        report["gaps"] = precision_recall_f1(pred_gaps, bench["ground_gaps"])

    # Relation eval
    if "ground_relations" in bench:
        report["relations"] = {"note": "requires claim ID alignment — run via test suite"}

    logger.info("Bench results: %s", json.dumps(report))
    return report


def f1_score(precision: float, recall: float) -> float:
    """F1 = 2 * P * R / (P + R). Returns 0 if both are 0."""
    return round(2 * precision * recall / (precision + recall + 1e-8), 3)
