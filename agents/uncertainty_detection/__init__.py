from __future__ import annotations
from typing import Any

class UncertaintyDetector:
    """Estimate uncertainty per claim from conflict/inconclusive evidence signals."""

    def detect(self, results: list[dict[str, Any]], evidence: list[dict[str, Any]] = None) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []

        for item in results:
            supporting = item.get("supporting", []) or []
            contradicting = item.get("contradicting", []) or []
            inconclusive = item.get("inconclusive", []) or []

            total = len(supporting) + len(contradicting) + len(inconclusive)
            uncertainty_score = round(((len(contradicting) + len(inconclusive)) / total) * 10, 2) if total else 5.0

            if uncertainty_score >= 7:
                level = "high"
            elif uncertainty_score >= 4:
                level = "medium"
            else:
                level = "low"

            output.append({
                "paper_id": item.get("focal_paper_id"),
                "title": item.get("focal_paper_title", ""),
                "claim": item.get("claim", ""),
                "uncertainty_score": uncertainty_score,
                "uncertainty_level": level,
                "inconclusive_count": len(inconclusive),
            })

        return output

__all__ = ["UncertaintyDetector"]
