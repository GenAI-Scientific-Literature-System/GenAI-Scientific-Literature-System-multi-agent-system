"""
Gap Ranker & Scoring Engine
============================
Advanced gap ranking with multi-dimensional scoring and clustering.
"""

import re
import math
from typing import List, Dict, Tuple
from collections import defaultdict


# ─────────────────────────────────────────────
# Impact domain keywords → multiplier
# ─────────────────────────────────────────────

IMPACT_KEYWORDS = {
    "clinical|medical|health|disease|patient|drug": 1.2,
    "safety|security|privacy|bias|fairness|harm": 1.2,
    "climate|environment|sustainable|energy": 1.15,
    "benchmark|evaluation|standard|dataset": 1.1,
    "scalab|large.scale|production|deploy": 1.1,
    "replicate|reproduce|replication": 1.05,
}

FEASIBILITY_BOOST_KEYWORDS = {
    "existing (data|model|framework|tool|library)": 0.15,
    "straightforward|simple|easy|incremental": 0.1,
    "open.source|publicly available|available": 0.1,
    "well.studied|established|mature": 0.05,
}

FEASIBILITY_PENALTY_KEYWORDS = {
    "require (significant|extensive|major) (effort|resource|data|funding)": -0.15,
    "fundamental|paradigm.shift|new theory": -0.1,
    "decades|long.term|many years": -0.12,
    "proprietary|private|restricted|confidential": -0.1,
}


def refine_impact_score(gap: Dict) -> float:
    """Apply domain-specific impact multipliers."""
    base = gap.get("impact_score", 0.7)
    text = (gap.get("description", "") + " " + gap.get("category", "")).lower()

    multiplier = 1.0
    for pattern, mult in IMPACT_KEYWORDS.items():
        if re.search(pattern, text):
            multiplier = max(multiplier, mult)

    return round(min(base * multiplier, 1.0), 3)


def refine_feasibility_score(gap: Dict) -> float:
    """Apply contextual feasibility adjustments."""
    base = gap.get("feasibility_score", 0.6)
    text = gap.get("description", "").lower()

    adjustment = 0.0
    for pattern, boost in FEASIBILITY_BOOST_KEYWORDS.items():
        if re.search(pattern, text):
            adjustment += boost
    for pattern, penalty in FEASIBILITY_PENALTY_KEYWORDS.items():
        if re.search(pattern, text):
            adjustment += penalty

    return round(min(max(base + adjustment, 0.1), 1.0), 3)


def compute_novelty_score(gap: Dict, all_gaps: List[Dict]) -> float:
    """
    Novelty = inverse frequency of this gap type + source uniqueness.
    """
    gap_type = gap.get("gap_type", "")
    source = gap.get("paper_source", "")

    # Count how many gaps of same type exist
    type_count = sum(1 for g in all_gaps if g.get("gap_type") == gap_type)
    # Count how many gaps from same paper
    source_count = sum(1 for g in all_gaps if g.get("paper_source") == source)

    type_novelty = 1.0 / math.log(type_count + 2)  # log dampening
    source_novelty = 1.0 / math.log(source_count + 2)

    return round((type_novelty * 0.6 + source_novelty * 0.4), 3)


def rank_gaps(gaps: List[Dict]) -> List[Dict]:
    """
    Full ranking pipeline:
    1. Refine impact & feasibility
    2. Compute novelty
    3. Compute composite
    4. Assign rank & tier
    """
    if not gaps:
        return []

    enriched = []
    for gap in gaps:
        g = dict(gap)
        g["impact_score"] = refine_impact_score(g)
        g["feasibility_score"] = refine_feasibility_score(g)
        g["novelty_score"] = compute_novelty_score(g, gaps)
        g["composite_score"] = round(
            g["impact_score"] * 0.45 +
            g["feasibility_score"] * 0.30 +
            g["novelty_score"] * 0.25,
            4
        )
        enriched.append(g)

    # Sort by composite
    enriched.sort(key=lambda x: -x["composite_score"])

    # Assign rank and tier
    for i, g in enumerate(enriched):
        g["rank"] = i + 1
        score = g["composite_score"]
        g["tier"] = "🔴 Critical" if score > 0.80 else \
                    "🟠 High" if score > 0.70 else \
                    "🟡 Medium" if score > 0.55 else \
                    "🟢 Low"
        g["priority"] = "HIGH" if score > 0.70 else "MEDIUM" if score > 0.55 else "LOW"

    return enriched


# ─────────────────────────────────────────────
# Gap clustering by theme
# ─────────────────────────────────────────────

THEME_PATTERNS = {
    "Data & Benchmarks": ["data", "dataset", "benchmark", "corpus", "sample", "annotation"],
    "Methodology": ["method", "model", "architecture", "algorithm", "approach", "framework"],
    "Evaluation": ["evaluat", "metric", "measure", "baseline", "comparison", "validation"],
    "Generalizability": ["general", "domain", "transfer", "robust", "universal", "diverse"],
    "Scalability": ["scale", "large", "production", "efficiency", "deploy", "real-world"],
    "Interpretability": ["interpret", "explain", "transparent", "understand", "visuali"],
    "Ethics & Safety": ["bias", "fair", "safe", "harm", "privacy", "ethic", "toxic"],
    "Theoretical": ["theory", "proof", "bound", "formal", "theorem", "principle"],
    "Applications": ["appli", "use case", "practial", "downstream", "task-specific"],
}


def cluster_gaps_by_theme(gaps: List[Dict]) -> Dict[str, List[Dict]]:
    """Group gaps by research theme."""
    clusters = defaultdict(list)

    for gap in gaps:
        text = (gap.get("description", "") + " " + gap.get("gap_type", "") + 
                " " + gap.get("category", "")).lower()
        
        assigned = False
        for theme, keywords in THEME_PATTERNS.items():
            if any(kw in text for kw in keywords):
                clusters[theme].append(gap)
                assigned = True
                break

        if not assigned:
            clusters["Other"].append(gap)

    # Sort each cluster by composite score
    return {theme: sorted(gs, key=lambda x: -x.get("composite_score", 0))
            for theme, gs in clusters.items() if gs}


def generate_gap_summary(gaps: List[Dict], paper_analyses: Dict) -> Dict:
    """Generate a human-readable summary of the gap landscape."""
    if not gaps:
        return {"summary": "No research gaps detected.", "key_findings": []}

    top_gaps = gaps[:5]
    critical = [g for g in gaps if g.get("tier", "").startswith("🔴")]
    
    themes = cluster_gaps_by_theme(gaps)
    top_theme = max(themes.items(), key=lambda x: len(x[1]))[0] if themes else "N/A"

    key_findings = []
    if critical:
        key_findings.append(f"{len(critical)} critical research gaps require immediate attention.")
    if top_theme != "N/A":
        key_findings.append(f"The '{top_theme}' cluster dominates the gap landscape.")
    
    avg_impact = sum(g.get("impact_score", 0) for g in gaps) / max(len(gaps), 1)
    avg_feasibility = sum(g.get("feasibility_score", 0) for g in gaps) / max(len(gaps), 1)
    
    key_findings.append(
        f"Average gap impact: {avg_impact:.1%}, average feasibility: {avg_feasibility:.1%}."
    )
    
    if len(paper_analyses) > 1:
        most_uncertain = max(
            paper_analyses.items(),
            key=lambda x: x[1].get("avg_uncertainty", 0)
        )
        key_findings.append(
            f"'{most_uncertain[0]}' is the most uncertain paper "
            f"(avg uncertainty: {most_uncertain[1].get('avg_uncertainty', 0):.1%})."
        )

    return {
        "total_gaps": len(gaps),
        "critical_gaps": len(critical),
        "top_themes": list(themes.keys())[:4],
        "avg_impact": round(avg_impact, 3),
        "avg_feasibility": round(avg_feasibility, 3),
        "key_findings": key_findings,
        "top_gap": top_gaps[0] if top_gaps else None
    }


if __name__ == "__main__":
    sample_gaps = [
        {
            "gap_id": "G001", "paper_source": "Paper_A",
            "gap_type": "data_gap", "category": "Missing Data / Benchmark",
            "description": "The dataset used lacks clinical diversity for medical NLP.",
            "impact_score": 0.8, "feasibility_score": 0.7, "novelty_score": 0.75,
            "composite_score": 0.755
        },
        {
            "gap_id": "G002", "paper_source": "Paper_B",
            "gap_type": "future_direction", "category": "Future Research Direction",
            "description": "Future work should explore scaling to billion-parameter models.",
            "impact_score": 0.75, "feasibility_score": 0.5, "novelty_score": 0.8,
            "composite_score": 0.68
        }
    ]
    ranked = rank_gaps(sample_gaps)
    for g in ranked:
        print(f"[{g['rank']}] {g['tier']} | {g['category']} | score={g['composite_score']:.3f}")
    
    clusters = cluster_gaps_by_theme(sample_gaps)
    print(f"\nTheme clusters: {list(clusters.keys())}")
