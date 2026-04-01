def build_consensus_conflicts_view(result: dict) -> dict:
    agreements = result.get("agreement_results", [])
    uncertainties = result.get("uncertainty_results", [])
    evidence_results = result.get("results", [])

    consensus_items = []
    conflict_items = []

    if isinstance(agreements, list):
        for item in agreements:
            if isinstance(item, dict):
                consensus_items.append({
                    "claim": item.get("claim", ""),
                    "status": item.get("status", "agreement"),
                    "papers": item.get("papers", []),
                    "summary": item.get("summary", "")
                })
            else:
                consensus_items.append({
                    "claim": str(item),
                    "status": "agreement",
                    "papers": [],
                    "summary": ""
                })

    if isinstance(uncertainties, list):
        for item in uncertainties:
            if isinstance(item, dict):
                conflict_items.append({
                    "claim": item.get("claim", ""),
                    "status": item.get("status", "conflict"),
                    "papers": item.get("papers", []),
                    "summary": item.get("summary", item.get("reason", ""))
                })
            else:
                conflict_items.append({
                    "claim": str(item),
                    "status": "conflict",
                    "papers": [],
                    "summary": ""
                })

    if not consensus_items and evidence_results:
        for item in evidence_results[:10]:
            support_count = len(item.get("supporting", []))
            contradict_count = len(item.get("contradicting", []))

            if support_count > contradict_count and support_count > 0:
                consensus_items.append({
                    "claim": item.get("claim", ""),
                    "status": "agreement",
                    "papers": [item.get("focal_paper_id", "")],
                    "summary": f"{support_count} supporting papers found"
                })
            elif contradict_count > 0:
                conflict_items.append({
                    "claim": item.get("claim", ""),
                    "status": "conflict",
                    "papers": [item.get("focal_paper_id", "")],
                    "summary": f"{contradict_count} contradicting papers found"
                })

    return {
        "consensus": consensus_items,
        "conflicts": conflict_items,
        "consensus_total": len(consensus_items),
        "conflicts_total": len(conflict_items)
    }