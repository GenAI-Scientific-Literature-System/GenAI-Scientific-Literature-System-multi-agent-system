from typing import Dict, Any, List


def compute_quality_metrics(result: Dict[str, Any]) -> Dict[str, Any]:
    claims = result.get("claims", [])
    evidence = result.get("evidence", [])
    reliability = result.get("reliability", [])
    ranked_insights = result.get("ranked_insights", [])
    errors = result.get("errors", [])

    return {
        "num_claims": len(claims) if isinstance(claims, list) else 0,
        "num_evidence_items": len(evidence) if isinstance(evidence, list) else 0,
        "num_reliability_entries": len(reliability) if isinstance(reliability, list) else 0,
        "num_ranked_insights": len(ranked_insights) if isinstance(ranked_insights, list) else 0,
        "error_count": len(errors)
    }


def precision_at_k(predicted: List[Any], ground_truth: List[Any], k: int = 5) -> float:
    predicted_k = predicted[:k]
    if not predicted_k:
        return 0.0

    correct = sum(1 for item in predicted_k if item in ground_truth)
    return round(correct / len(predicted_k), 4)