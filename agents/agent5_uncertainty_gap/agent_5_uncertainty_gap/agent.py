"""
Agent 5: Uncertainty & Research Gap Analyst
=============================================
Detects ambiguity, weak evidence, conflicting conclusions.
Identifies open research questions, missing experiments, underexplored directions.
Ranks gaps by Impact × Feasibility × Novelty.
"""

import re
import json
import math
from typing import List, Dict, Tuple
from collections import defaultdict


# ─────────────────────────────────────────────
# Uncertainty Detection Lexicons
# ─────────────────────────────────────────────

HEDGING_PATTERNS = [
    (r"\b(may|might|could|possibly|perhaps|probably|likely|seemingly|apparently)\b", 0.7, "epistemic_hedge"),
    (r"\b(suggest|indicate|appear|seem|imply)\b", 0.65, "soft_claim"),
    (r"\b(limited|preliminary|pilot|exploratory|initial)\b", 0.6, "scope_limit"),
    (r"\b(further (work|study|research|investigation) (is|are)? (needed|required|warranted))\b", 0.8, "gap_signal"),
    (r"\b(unclear|unknown|uncertain|ambiguous|inconclusive)\b", 0.85, "explicit_uncertainty"),
    (r"\b(to the best of our knowledge|as far as we know)\b", 0.75, "knowledge_boundary"),
    (r"\b(we assume|we hypothesize|we conjecture|we speculate)\b", 0.7, "assumption"),
    (r"\b(small (sample|dataset|cohort|set))\b", 0.65, "data_weakness"),
    (r"\b(not (yet|fully|completely) (understood|explored|validated|tested))\b", 0.8, "incomplete_knowledge"),
    (r"\b(more (data|evidence|experiments|studies) (is|are)? needed)\b", 0.85, "evidence_gap"),
    (r"\b(future work|future research|future studies)\b", 0.75, "future_work"),
    (r"\b(beyond (the scope|this (paper|work|study)))\b", 0.7, "scope_exclusion"),
    (r"\b(cannot (be|fully) (generalized|applied|extended))\b", 0.75, "generalizability_limit"),
    (r"\b(we do not (address|consider|include|evaluate))\b", 0.7, "omission"),
    (r"\b(limitation|drawback|weakness|shortcoming|caveat)\b", 0.8, "limitation"),
]

STRONG_CLAIM_PATTERNS = [
    r"\b(definitively|conclusively|unambiguously|clearly|undoubtedly)\b",
    r"\b(always|never|all|none|every|no|universal)\b",
    r"\b(proves?|establishes?|confirms?)\b",
    r"\b(state-of-the-art|best|optimal|perfect|ideal)\b",
]

METHODOLOGY_WEAKNESS_PATTERNS = [
    (r"\b(n\s*=\s*\d{1,2})\b", 0.7, "small_n"),
    (r"\b(\d{1,2}\s+subjects?|\d{1,2}\s+participants?)\b", 0.65, "small_sample"),
    (r"\b(single (study|paper|experiment|dataset))\b", 0.7, "single_source"),
    (r"\b(not (peer-reviewed|validated|replicated))\b", 0.75, "validation_missing"),
    (r"\b(synthetic|simulated|toy|artificial) (data|dataset|examples?)\b", 0.65, "synthetic_data"),
    (r"\b(ablation study (not|was not|did not))\b", 0.6, "missing_ablation"),
    (r"\b(no (baseline|comparison|control|ablation))\b", 0.7, "missing_baseline"),
    (r"\b(self-reported|subjective|qualitative (only|assessment))\b", 0.6, "subjective_method"),
]

RESEARCH_GAP_TRIGGERS = [
    (r"\b(future work|future research|future studies)\b", "future_direction"),
    (r"\b(open (question|problem|challenge|issue))\b", "open_question"),
    (r"\b(remains? (unknown|unexplored|understudied|open))\b", "unexplored"),
    (r"\b(has not been (studied|explored|investigated|addressed))\b", "unstudied"),
    (r"\b(lack (of|ing) (data|evidence|benchmarks?|datasets?))\b", "data_gap"),
    (r"\b(further (investigation|analysis|study) (is|are)? (needed|required))\b", "investigation_needed"),
    (r"\b(we leave (this|the) .{5,50} (for|to) future)\b", "deferred_work"),
    (r"\b(promising direction|potential (avenue|direction|extension))\b", "promising_direction"),
    (r"\b(underexplored|understudied|overlooked|neglected)\b", "underexplored"),
    (r"\b(broader (impact|implications|applications) (are|remain) (unclear|unknown))\b", "broad_impact_gap"),
]


# ─────────────────────────────────────────────
# Sentence-level uncertainty scoring
# ─────────────────────────────────────────────

def score_sentence_uncertainty(sentence: str) -> Tuple[float, List[str], List[str]]:
    """
    Returns (uncertainty_score, matched_types, matched_spans).
    """
    s = sentence.lower()
    hits = []
    spans = []
    total_score = 0.0
    count = 0

    for pattern, weight, label in HEDGING_PATTERNS:
        m = re.search(pattern, s)
        if m:
            hits.append(label)
            spans.append(m.group(0))
            total_score += weight
            count += 1

    for pattern, weight, label in METHODOLOGY_WEAKNESS_PATTERNS:
        m = re.search(pattern, s)
        if m:
            hits.append(label)
            spans.append(m.group(0))
            total_score += weight
            count += 1

    # Strong claims reduce uncertainty
    strong_count = sum(
        1 for p in STRONG_CLAIM_PATTERNS if re.search(p, s)
    )
    total_score = max(0.0, total_score - strong_count * 0.15)

    if count == 0:
        return 0.0, [], []

    normalized = min(total_score / (count + 1), 1.0)
    return round(normalized, 3), hits, spans


def detect_gap_signals(sentence: str) -> List[Dict]:
    """Extract gap signals from a sentence."""
    s = sentence.lower()
    gaps = []
    for pattern, gap_type in RESEARCH_GAP_TRIGGERS:
        m = re.search(pattern, s)
        if m:
            gaps.append({
                "type": gap_type,
                "trigger": m.group(0),
                "sentence": sentence.strip()
            })
    return gaps


# ─────────────────────────────────────────────
# Paper-level analysis
# ─────────────────────────────────────────────

def analyze_paper_uncertainty(paper_id: str, text: str) -> Dict:
    """
    Full uncertainty analysis for a single paper.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 25]

    uncertain_sentences = []
    gap_signals = []
    all_uncertainty_types = defaultdict(int)
    total_uncertainty = 0.0

    for sent in sentences:
        u_score, u_types, u_spans = score_sentence_uncertainty(sent)
        if u_score > 0.1:
            uncertain_sentences.append({
                "sentence": sent,
                "uncertainty_score": u_score,
                "uncertainty_types": u_types,
                "uncertainty_spans": u_spans
            })
            total_uncertainty += u_score
            for t in u_types:
                all_uncertainty_types[t] += 1

        gaps = detect_gap_signals(sent)
        gap_signals.extend(gaps)

    avg_uncertainty = total_uncertainty / max(len(sentences), 1)

    # Deduplicate gap signals by type+trigger
    seen_gaps = set()
    unique_gaps = []
    for g in gap_signals:
        key = (g["type"], g["trigger"])
        if key not in seen_gaps:
            seen_gaps.add(key)
            unique_gaps.append(g)

    return {
        "paper_id": paper_id,
        "total_sentences": len(sentences),
        "uncertain_sentence_count": len(uncertain_sentences),
        "avg_uncertainty": round(avg_uncertainty, 4),
        "uncertainty_ratio": round(len(uncertain_sentences) / max(len(sentences), 1), 3),
        "dominant_uncertainty_types": dict(sorted(all_uncertainty_types.items(), key=lambda x: -x[1])[:5]),
        "uncertain_sentences": sorted(uncertain_sentences, key=lambda x: -x["uncertainty_score"])[:8],
        "gap_signals": unique_gaps[:10]
    }


# ─────────────────────────────────────────────
# Research Gap Generation
# ─────────────────────────────────────────────

GAP_CATEGORIES = {
    "future_direction": ("Future Research Direction", 0.9, 0.8),
    "open_question":    ("Open Scientific Question", 0.95, 0.6),
    "unexplored":       ("Unexplored Area", 0.85, 0.7),
    "unstudied":        ("Unstudied Problem", 0.9, 0.65),
    "data_gap":         ("Missing Data / Benchmark", 0.8, 0.75),
    "investigation_needed": ("Investigation Required", 0.85, 0.8),
    "deferred_work":    ("Deferred by Authors", 0.75, 0.85),
    "promising_direction": ("Promising Direction", 0.7, 0.9),
    "underexplored":    ("Underexplored Domain", 0.8, 0.7),
    "broad_impact_gap": ("Broader Impact Unknown", 0.7, 0.5),
    "epistemic_hedge":  ("Epistemic Uncertainty", 0.6, 0.6),
    "data_weakness":    ("Weak Empirical Evidence", 0.75, 0.7),
    "small_n":          ("Insufficient Sample Size", 0.85, 0.9),
    "missing_baseline": ("Missing Baseline Comparison", 0.8, 0.85),
    "validation_missing": ("External Validation Missing", 0.85, 0.8),
    "generalizability_limit": ("Generalizability Unknown", 0.9, 0.65),
    "limitation":       ("Acknowledged Limitation", 0.7, 0.75),
}


def generate_research_gaps(paper_analyses: Dict[str, Dict]) -> List[Dict]:
    """
    Synthesize research gaps from all paper analyses.
    Rank by Impact × Feasibility × Novelty composite score.
    """
    gaps = []
    seen = set()

    for paper_id, analysis in paper_analyses.items():
        for gap_signal in analysis.get("gap_signals", []):
            gap_type = gap_signal["type"]
            sentence = gap_signal["sentence"]
            trigger = gap_signal["trigger"]

            key = (gap_type, sentence[:60])
            if key in seen:
                continue
            seen.add(key)

            cat_info = GAP_CATEGORIES.get(gap_type, ("Research Gap", 0.7, 0.6))
            category, impact, feasibility = cat_info

            # Novelty: unique to this paper = higher novelty
            novelty = 0.8

            composite = round((impact * 0.45 + feasibility * 0.3 + novelty * 0.25), 3)

            recommendation = _generate_recommendation(gap_type, sentence, paper_id)

            gaps.append({
                "gap_id": f"G{len(gaps)+1:03d}",
                "paper_source": paper_id,
                "gap_type": gap_type,
                "category": category,
                "description": sentence,
                "trigger_phrase": trigger,
                "impact_score": impact,
                "feasibility_score": feasibility,
                "novelty_score": novelty,
                "composite_score": composite,
                "recommendation": recommendation,
                "priority": "HIGH" if composite > 0.75 else "MEDIUM" if composite > 0.55 else "LOW"
            })

        # Gaps from uncertain sentences
        for usent in analysis.get("uncertain_sentences", [])[:3]:
            for utype in usent.get("uncertainty_types", []):
                key = (utype, usent["sentence"][:60])
                if key in seen:
                    continue
                seen.add(key)

                cat_info = GAP_CATEGORIES.get(utype, ("Research Gap", 0.6, 0.6))
                category, impact, feasibility = cat_info
                novelty = 0.65
                composite = round((impact * 0.45 + feasibility * 0.3 + novelty * 0.25), 3)

                gaps.append({
                    "gap_id": f"G{len(gaps)+1:03d}",
                    "paper_source": paper_id,
                    "gap_type": utype,
                    "category": category,
                    "description": usent["sentence"],
                    "trigger_phrase": ", ".join(usent.get("uncertainty_spans", [])[:2]),
                    "impact_score": impact,
                    "feasibility_score": feasibility,
                    "novelty_score": novelty,
                    "composite_score": composite,
                    "recommendation": _generate_recommendation(utype, usent["sentence"], paper_id),
                    "priority": "HIGH" if composite > 0.75 else "MEDIUM" if composite > 0.55 else "LOW"
                })

    # Sort by composite score
    gaps.sort(key=lambda x: -x["composite_score"])
    return gaps


def _generate_recommendation(gap_type: str, sentence: str, paper_id: str) -> str:
    """Generate an actionable recommendation for a research gap."""
    recs = {
        "future_direction": f"Design a follow-up study extending the work from {paper_id}. Formulate a testable hypothesis based on the stated direction.",
        "open_question": f"Formulate this open question as a falsifiable hypothesis. Design controlled experiments to address it.",
        "unexplored": f"Conduct a systematic survey of the unexplored area identified in {paper_id}. Begin with a literature review + pilot study.",
        "unstudied": f"This problem lacks existing literature. A first-of-its-kind empirical study from {paper_id}'s domain would be high-impact.",
        "data_gap": f"Curate or collect a dedicated benchmark dataset addressing the gap in {paper_id}. Establish evaluation protocols.",
        "investigation_needed": f"Design controlled experiments with proper baselines to investigate the open question in {paper_id}.",
        "deferred_work": f"The authors of {paper_id} deferred this work. It represents a natural extension with a clear starting point.",
        "promising_direction": f"Develop a prototype exploring this direction from {paper_id}. Evaluate against current baselines.",
        "underexplored": f"A systematic study of this underexplored area would provide foundational knowledge for the field.",
        "broad_impact_gap": f"Conduct an impact assessment of the broader implications of {paper_id}'s findings across domains.",
        "epistemic_hedge": f"Re-examine the hedged claims in {paper_id} with stronger experimental designs and larger samples.",
        "data_weakness": f"Replicate {paper_id}'s study with larger, more diverse datasets to validate the findings.",
        "small_n": f"Reproduce {paper_id}'s experiments at scale. Power analysis suggests minimum N for reliable conclusions.",
        "missing_baseline": f"Add missing baselines to {paper_id}'s evaluation. Compare against established state-of-the-art.",
        "validation_missing": f"Conduct external validation of {paper_id}'s claims on independent datasets.",
        "generalizability_limit": f"Test {paper_id}'s approach across diverse domains, languages, datasets to establish generalizability.",
        "limitation": f"Address the acknowledged limitations in {paper_id} directly in future work.",
    }
    return recs.get(gap_type, f"Investigate the research gap identified in {paper_id} systematically.")


# ─────────────────────────────────────────────
# Cross-paper conflict detection
# ─────────────────────────────────────────────

def detect_cross_paper_conflicts(paper_analyses: Dict[str, Dict]) -> List[Dict]:
    """
    Detect conflicting uncertainty patterns across papers.
    E.g., one paper claims certainty where another shows high uncertainty.
    """
    conflicts = []
    papers = list(paper_analyses.keys())

    for i in range(len(papers)):
        for j in range(i + 1, len(papers)):
            p1, p2 = papers[i], papers[j]
            a1 = paper_analyses[p1]
            a2 = paper_analyses[p2]

            u1 = a1.get("avg_uncertainty", 0)
            u2 = a2.get("avg_uncertainty", 0)
            diff = abs(u1 - u2)

            if diff > 0.08:
                more_uncertain = p1 if u1 > u2 else p2
                less_uncertain = p2 if u1 > u2 else p1
                conflicts.append({
                    "paper_1": p1,
                    "paper_2": p2,
                    "paper_1_uncertainty": u1,
                    "paper_2_uncertainty": u2,
                    "uncertainty_gap": round(diff, 4),
                    "interpretation": (
                        f"{more_uncertain} expresses significantly higher uncertainty "
                        f"than {less_uncertain} on overlapping topics. "
                        "This may indicate methodological differences, data limitations, "
                        "or genuine scientific disagreement about confidence levels."
                    ),
                    "severity": "HIGH" if diff > 0.2 else "MEDIUM"
                })

    return sorted(conflicts, key=lambda x: -x["uncertainty_gap"])


# ─────────────────────────────────────────────
# Main Agent 5 Pipeline
# ─────────────────────────────────────────────

def run_agent_5(papers: Dict[str, str]) -> Dict:
    """
    Full Agent 5 pipeline:
    1. Analyze uncertainty per paper
    2. Detect gap signals
    3. Generate ranked research gaps
    4. Detect cross-paper conflicts
    5. Aggregate statistics
    """
    print("[Agent 5] Analyzing uncertainty and research gaps...")
    paper_analyses = {}

    for paper_id, text in papers.items():
        analysis = analyze_paper_uncertainty(paper_id, text)
        paper_analyses[paper_id] = analysis
        print(f"  → {paper_id}: avg_uncertainty={analysis['avg_uncertainty']:.3f}, "
              f"gaps={len(analysis['gap_signals'])}, "
              f"uncertain_sents={analysis['uncertain_sentence_count']}")

    print("[Agent 5] Generating ranked research gaps...")
    gaps = generate_research_gaps(paper_analyses)

    print("[Agent 5] Detecting cross-paper conflicts...")
    conflicts = detect_cross_paper_conflicts(paper_analyses)

    # Aggregate summary
    all_uncertainties = [a["avg_uncertainty"] for a in paper_analyses.values()]
    avg_uncertainty_overall = sum(all_uncertainties) / max(len(all_uncertainties), 1)

    gap_priority_counts = defaultdict(int)
    for g in gaps:
        gap_priority_counts[g["priority"]] += 1

    gap_type_distribution = defaultdict(int)
    for g in gaps:
        gap_type_distribution[g["gap_type"]] += 1

    summary = {
        "total_papers": len(papers),
        "total_gaps": len(gaps),
        "high_priority_gaps": gap_priority_counts["HIGH"],
        "medium_priority_gaps": gap_priority_counts["MEDIUM"],
        "low_priority_gaps": gap_priority_counts["LOW"],
        "avg_uncertainty_overall": round(avg_uncertainty_overall, 4),
        "cross_paper_conflicts": len(conflicts),
        "top_gap_types": dict(sorted(gap_type_distribution.items(), key=lambda x: -x[1])[:5]),
    }

    print(f"[Agent 5] Done. {len(gaps)} gaps | {summary['high_priority_gaps']} HIGH | "
          f"{len(conflicts)} conflicts | avg_uncertainty={avg_uncertainty_overall:.3f}")

    return {
        "agent": "Agent_5_Uncertainty_Gap",
        "paper_analyses": paper_analyses,
        "research_gaps": gaps,
        "cross_paper_conflicts": conflicts,
        "summary": summary
    }


if __name__ == "__main__":
    sample_papers = {
        "Paper_Alpha": (
            "We suggest that transformer models may improve performance in clinical NLP tasks. "
            "However, our study is limited by a small sample size (n=12 patients). "
            "Future work should explore larger and more diverse datasets. "
            "The generalizability of these findings to other medical domains remains unclear. "
            "We assume that the data is representative, but this has not been validated. "
            "Further investigation is needed to confirm these preliminary results."
        ),
        "Paper_Beta": (
            "Our model definitively outperforms all baselines on standard benchmarks. "
            "We conclusively demonstrate that attention mechanisms are crucial. "
            "The results are universally applicable across all NLP tasks. "
            "We prove that our approach is optimal in all settings. "
            "Future research directions include scaling to larger models. "
            "Open questions remain about low-resource language adaptation."
        )
    }
    result = run_agent_5(sample_papers)
    print(json.dumps(result["summary"], indent=2))
    print(f"\nTop 3 gaps:")
    for g in result["research_gaps"][:3]:
        print(f"  [{g['priority']}] {g['category']}: {g['description'][:80]}...")
