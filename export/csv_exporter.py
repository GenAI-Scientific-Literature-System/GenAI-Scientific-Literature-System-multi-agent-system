import csv
import os
from datetime import datetime


def export_ranked_insights_csv(result: dict) -> str:
    os.makedirs("exports", exist_ok=True)

    filename = f"exports/ranked_insights_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    ranked_insights = result.get("ranked_insights", [])

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "insight", "score", "source_paper", "type"])

        for idx, item in enumerate(ranked_insights, start=1):
            if isinstance(item, dict):
                writer.writerow([
                    idx,
                    item.get("insight", item.get("claim", "")),
                    item.get("score", ""),
                    item.get("source_paper", ""),
                    item.get("type", "")
                ])
            else:
                writer.writerow([idx, str(item), "", "", ""])

    return filename