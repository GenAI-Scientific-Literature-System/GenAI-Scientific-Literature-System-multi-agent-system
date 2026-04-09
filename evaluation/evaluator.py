from typing import Dict, Any, List

from evaluation.metrics import compute_quality_metrics, precision_at_k
from monitoring.metrics import compute_system_metrics


class Evaluator:
    def evaluate(self, result: Dict[str, Any], expected_topics: List[str] = None) -> Dict[str, Any]:
        expected_topics = expected_topics or []

        quality_metrics = compute_quality_metrics(result)
        system_metrics = compute_system_metrics(result)

        ranked_insights = result.get("ranked_insights", [])
        extracted_texts = []

        for item in ranked_insights:
            if isinstance(item, dict):
                extracted_texts.append(item.get("insight", "") or item.get("claim", "") or item.get("title", ""))

        p_at_5 = precision_at_k(extracted_texts, expected_topics, k=5)

        return {
            "quality_metrics": quality_metrics,
            "system_metrics": system_metrics,
            "ranking_metrics": {
                "precision_at_5": p_at_5
            }
        }