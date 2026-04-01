def normalize(value, min_value = 0, max_value = 10):
    if value < min_value:
        value = min_value
    if value > max_value:
        value = max_value
    return (value - min_value) / (max_value - min_value) if max_value > min_value else 0.0

def compute_priority_score(insight):
    reliability = normalize(insight.get('reliability_score', 0), 0, 10)
    evidence = normalize(insight.get('evidence_count', 0), 0, 20)
    agreement = normalize(insight.get("agreement_score", 0), 0, 10)
    conflict = normalize(insight.get("conflict_score", 0), 0, 10)
    novelty = normalize(insight.get("novelty_score", 0), 0, 10)

    score = (
        0.30 * reliability +
        0.25 * evidence +
        0.20 * agreement +
        0.15 * novelty -
        0.10 * conflict
    )

    return round(score * 100, 2)

def build_reason(insight, score):
    reasons = []
    if insight.get("reliability_score", 0) >= 7:
        reasons.append("high source reliability")

    if insight.get("evidence_count", 0) >= 5:
        reasons.append("strong supporting evidence")

    if insight.get("agreement_score", 0) >= 7:
        reasons.append("strong consensus across papers")

    if insight.get("novelty_score", 0) >= 7:
        reasons.append("high novelty or importance")

    if insight.get("conflict_score", 0) >= 6:
        reasons.append("some contradiction present")

    if not reasons:
        reasons.append("moderate overall relevance")

    return f"Priority score {score} based on " + ", ".join(reasons) + "."

def rank_insights_heuristically(insights):
    ranked = []

    for insight in insights:
        score = compute_priority_score(insight)
        ranked.append({
            "paper_id": insight.get("paper_id", ""),
            "title": insight.get("title", ""),
            "claim": insight.get("claim", ""),
            "priority_score": score,
            "reason": build_reason(insight, score)
        })

    ranked.sort(key=lambda x: x["priority_score"], reverse=True)

    for i, item in enumerate(ranked, start=1):
        item["rank"] = i

    return ranked