def build_ranked_insights_view(result: dict) -> dict:
    ranked_insights = result.get("ranked_insights", [])
    agreements = result.get("agreement_results", [])
    uncertainties = result.get("uncertainty_results", [])

    top_claims = []
    for idx, item in enumerate(ranked_insights[:10], start=1):
        if isinstance(item, dict):
            top_claims.append({
                "rank": idx,
                "insight": item.get("insight", item.get("claim", "")),
                "score": item.get("score", 0),
                "source_paper": item.get("source_paper", ""),
                "type": item.get("type", "ranked_insight")
            })
        else:
            top_claims.append({
                "rank": idx,
                "insight": str(item),
                "score": 0,
                "source_paper": "",
                "type": "ranked_insight"
            })

    return {
        "top_claims": top_claims,
        "consensus_count": len(agreements) if isinstance(agreements, list) else 0,
        "conflict_count": len(uncertainties) if isinstance(uncertainties, list) else 0
    }