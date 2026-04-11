from __future__ import annotations
from typing import Any

class AgreementDetector:
    """Compute per-claim agreement/conflict signals from evidence buckets."""

    def detect(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        agreements: list[dict[str, Any]] = []

        for item in results:
            supporting = item.get("supporting", []) or []
            contradicting = item.get("contradicting", []) or []
            total = len(supporting) + len(contradicting)

            agreement_score = round((len(supporting) / total) * 10, 2) if total else 0.0
            conflict_score = round((len(contradicting) / total) * 10, 2) if total else 0.0

            agreements.append({
                "paper_id": item.get("focal_paper_id"),
                "title": item.get("focal_paper_title", ""),
                "claim": item.get("claim", ""),
                "supporting_count": len(supporting),
                "contradicting_count": len(contradicting),
                "agreement_score": agreement_score,
                "conflict_score": conflict_score,
            })

        return agreements

__all__ = ["AgreementDetector"]
