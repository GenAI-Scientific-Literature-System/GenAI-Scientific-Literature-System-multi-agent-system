"""
Agent 4: Agreement & Disagreement Analyst
==========================================
Extracts structured claims from papers, performs semantic comparison,
and classifies relationships: agreement, contradiction, partial, novel.
"""

import json
import re
import math
from typing import List, Dict, Tuple
from collections import defaultdict


# ─────────────────────────────────────────────
# Claim Extraction
# ─────────────────────────────────────────────

CLAIM_PATTERNS = [
    r"we (show|demonstrate|prove|find|observe|propose|present|establish|confirm|verify)[^.]+\.",
    r"(results|experiments|analysis|findings|study|data) (show|indicate|suggest|reveal|demonstrate)[^.]+\.",
    r"(our|the) (model|method|approach|framework|system|algorithm)[^.]+\.",
    r"(this|the) (paper|work|study|research)[^.]+\.",
    r"(we|our) (achieve|outperform|improve|increase|reduce|exceed|surpass)[^.]+\.",
    r"(it|this) (is|was|can be|has been) (shown|demonstrated|established|proven)[^.]+\.",
    r"(accuracy|performance|precision|recall|f1|auc|loss|error)[^.]*(%|percent|\d)[^.]+\.",
    r"(unlike|in contrast|compared to|while|whereas)[^.]+\.",
    r"(however|although|despite|nevertheless)[^.]+\.",
]

def extract_claims(paper_text: str, paper_id: str) -> List[Dict]:
    """Extract structured claims from paper text."""
    sentences = re.split(r'(?<=[.!?])\s+', paper_text)
    claims = []
    seen = set()

    for sent in sentences:
        sent_clean = sent.strip()
        if len(sent_clean) < 30 or sent_clean in seen:
            continue

        score = 0
        matched_pattern = None
        for pattern in CLAIM_PATTERNS:
            if re.search(pattern, sent_clean, re.IGNORECASE):
                score += 1
                if matched_pattern is None:
                    matched_pattern = pattern

        if score > 0:
            seen.add(sent_clean)
            claims.append({
                "claim_id": f"{paper_id}_c{len(claims)+1}",
                "paper_id": paper_id,
                "text": sent_clean,
                "claim_score": min(score / 3.0, 1.0),
                "pattern_matched": matched_pattern,
                "category": _categorize_claim(sent_clean)
            })

    return claims


def _categorize_claim(text: str) -> str:
    """Categorize a claim into a semantic type."""
    text_lower = text.lower()
    if any(w in text_lower for w in ["outperform", "improve", "better", "superior", "exceed", "surpass", "achieve"]):
        return "performance"
    elif any(w in text_lower for w in ["propose", "introduce", "present", "framework", "method", "model", "algorithm"]):
        return "methodology"
    elif any(w in text_lower for w in ["find", "observe", "discover", "reveal", "show", "demonstrate"]):
        return "finding"
    elif any(w in text_lower for w in ["however", "unlike", "contrast", "whereas", "but", "although"]):
        return "contrast"
    elif any(w in text_lower for w in ["limit", "fail", "cannot", "unable", "weakness", "drawback"]):
        return "limitation"
    else:
        return "general"


# ─────────────────────────────────────────────
# Semantic Similarity (TF-IDF cosine, no deps)
# ─────────────────────────────────────────────

def _tokenize(text: str) -> List[str]:
    return re.findall(r'\b[a-z]{3,}\b', text.lower())

def _tfidf_vectors(texts: List[str]) -> List[Dict[str, float]]:
    tokenized = [_tokenize(t) for t in texts]
    N = len(texts)
    df = defaultdict(int)
    for tokens in tokenized:
        for word in set(tokens):
            df[word] += 1
    vectors = []
    for tokens in tokenized:
        tf = defaultdict(float)
        for word in tokens:
            tf[word] += 1
        vec = {}
        for word, count in tf.items():
            tfidf = (count / len(tokens)) * math.log((N + 1) / (df[word] + 1))
            vec[word] = tfidf
        vectors.append(vec)
    return vectors

def cosine_similarity(v1: Dict[str, float], v2: Dict[str, float]) -> float:
    keys = set(v1) & set(v2)
    dot = sum(v1[k] * v2[k] for k in keys)
    mag1 = math.sqrt(sum(x**2 for x in v1.values()))
    mag2 = math.sqrt(sum(x**2 for x in v2.values()))
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot / (mag1 * mag2)

def semantic_similarity(text1: str, text2: str) -> float:
    vecs = _tfidf_vectors([text1, text2])
    return cosine_similarity(vecs[0], vecs[1])


# ─────────────────────────────────────────────
# Contradiction & Negation Detection
# ─────────────────────────────────────────────

NEGATION_WORDS = ["not", "no", "never", "neither", "nor", "fail", "cannot",
                  "doesn't", "don't", "didn't", "isn't", "aren't", "wasn't",
                  "opposite", "contrary", "unlike", "disagree", "reject", "disprove"]

CONTRAST_WORDS = ["however", "but", "although", "yet", "while", "whereas",
                  "in contrast", "on the other hand", "despite", "nevertheless",
                  "unlike", "counter", "instead", "rather than"]

def _has_negation(text: str) -> bool:
    t = text.lower()
    return any(word in t for word in NEGATION_WORDS)

def _has_contrast(text: str) -> bool:
    t = text.lower()
    return any(word in t for word in CONTRAST_WORDS)

def _extract_numeric_claims(text: str) -> List[float]:
    return [float(m) for m in re.findall(r'\b(\d+\.?\d*)\s*%?', text)]


# ─────────────────────────────────────────────
# Relationship Classifier
# ─────────────────────────────────────────────

def classify_relationship(claim1: Dict, claim2: Dict) -> Dict:
    """
    Compare two claims and classify their relationship.
    Returns a structured comparison result.
    """
    text1, text2 = claim1["text"], claim2["text"]
    sim = semantic_similarity(text1, text2)

    neg1, neg2 = _has_negation(text1), _has_negation(text2)
    contrast1, contrast2 = _has_contrast(text1), _has_contrast(text2)
    nums1, nums2 = _extract_numeric_claims(text1), _extract_numeric_claims(text2)

    # Numeric contradiction check
    numeric_conflict = False
    if nums1 and nums2:
        diff = abs(max(nums1) - max(nums2))
        if diff > 15:
            numeric_conflict = True

    # Category match
    same_category = claim1["category"] == claim2["category"]

    # Decision logic
    if sim < 0.08:
        relationship = "novel"
        confidence = 0.75 + (0.1 * claim1["claim_score"])
        explanation = (
            f"The claims address distinct topics with minimal lexical overlap (sim={sim:.2f}). "
            f"Claim 1 is a '{claim1['category']}' statement; Claim 2 is a '{claim2['category']}' statement. "
            "They represent independent contributions without direct overlap."
        )
    elif (neg1 != neg2 and sim > 0.15) or numeric_conflict or (contrast1 or contrast2):
        relationship = "contradiction"
        confidence = 0.6 + (0.2 * sim)
        explanation = (
            f"Potential contradiction detected (sim={sim:.2f}). "
            f"Negation polarity: Paper1={'negated' if neg1 else 'positive'}, "
            f"Paper2={'negated' if neg2 else 'positive'}. "
            f"{'Numeric values diverge significantly. ' if numeric_conflict else ''}"
            f"{'Contrast language detected. ' if contrast1 or contrast2 else ''}"
            "Both claims address similar subjects but reach opposing conclusions."
        )
    elif sim > 0.45 and same_category:
        relationship = "agreement"
        confidence = 0.65 + (0.3 * sim)
        explanation = (
            f"Strong semantic alignment (sim={sim:.2f}) between claims of the same category "
            f"('{claim1['category']}'). Both papers converge on similar findings, "
            "lending mutual evidential support."
        )
    elif sim > 0.18:
        relationship = "partial"
        confidence = 0.5 + (0.2 * sim)
        explanation = (
            f"Moderate topical overlap (sim={sim:.2f}) but differing scope or framing. "
            f"Claim categories: '{claim1['category']}' vs '{claim2['category']}'. "
            "The papers share a research area but differ in methodology, scope, or conclusions."
        )
    else:
        relationship = "novel"
        confidence = 0.55
        explanation = (
            f"Low similarity (sim={sim:.2f}) with no strong contradiction signals. "
            "This claim introduces a perspective not directly addressed by the other paper."
        )

    return {
        "claim": text1,
        "papers": [claim1["paper_id"], claim2["paper_id"]],
        "relationship": relationship,
        "confidence": round(min(confidence, 0.99), 3),
        "evidence": [text1[:200], text2[:200]],
        "uncertainty_score": round(1.0 - confidence, 3),
        "research_gap": _infer_gap(relationship, claim1, claim2),
        "explanation": explanation,
        "semantic_similarity": round(sim, 4),
        "claim_pair": {
            "claim1_id": claim1["claim_id"],
            "claim2_id": claim2["claim_id"],
            "claim1_category": claim1["category"],
            "claim2_category": claim2["category"]
        }
    }


def _infer_gap(relationship: str, c1: Dict, c2: Dict) -> str:
    if relationship == "contradiction":
        return (
            f"Empirical study needed to resolve conflicting claims about "
            f"'{c1['category']}' between {c1['paper_id']} and {c2['paper_id']}. "
            "Controlled experiments with shared benchmarks would clarify the discrepancy."
        )
    elif relationship == "partial":
        return (
            f"Further research could bridge the gap between '{c1['category']}' and "
            f"'{c2['category']}' perspectives. A unified framework integrating both views is missing."
        )
    elif relationship == "novel":
        return (
            f"The novel claim from {c1['paper_id']} on '{c1['category']}' lacks corroboration. "
            "Independent replication and cross-domain validation are needed."
        )
    else:
        return (
            f"While both papers agree on '{c1['category']}', the underlying mechanisms "
            "and generalizability to other domains remain underexplored."
        )


# ─────────────────────────────────────────────
# Main Analysis Runner
# ─────────────────────────────────────────────

def run_agent_4(papers: Dict[str, str]) -> Dict:
    """
    Full Agent 4 pipeline:
    1. Extract claims from all papers
    2. Cross-compare all claim pairs
    3. Classify relationships
    4. Aggregate statistics
    """
    print("[Agent 4] Extracting claims from papers...")
    all_claims = {}
    for paper_id, text in papers.items():
        claims = extract_claims(text, paper_id)
        all_claims[paper_id] = claims
        print(f"  → {paper_id}: {len(claims)} claims extracted")

    print("[Agent 4] Comparing claims across papers...")
    comparisons = []
    paper_ids = list(all_claims.keys())

    for i in range(len(paper_ids)):
        for j in range(i + 1, len(paper_ids)):
            pid1, pid2 = paper_ids[i], paper_ids[j]
            claims1 = all_claims[pid1]
            claims2 = all_claims[pid2]

            # Compare top N claims to keep it tractable
            for c1 in claims1[:10]:
                best_sim = 0
                best_result = None
                for c2 in claims2[:10]:
                    result = classify_relationship(c1, c2)
                    if result["confidence"] > best_sim:
                        best_sim = result["confidence"]
                        best_result = result
                if best_result:
                    comparisons.append(best_result)

    # Aggregate statistics
    stats = defaultdict(int)
    for comp in comparisons:
        stats[comp["relationship"]] += 1

    total = len(comparisons) or 1
    summary = {
        "total_comparisons": len(comparisons),
        "agreement_count": stats["agreement"],
        "contradiction_count": stats["contradiction"],
        "partial_count": stats["partial"],
        "novel_count": stats["novel"],
        "agreement_ratio": round(stats["agreement"] / total, 3),
        "contradiction_ratio": round(stats["contradiction"] / total, 3),
        "avg_confidence": round(
            sum(c["confidence"] for c in comparisons) / total, 3
        ),
    }

    print(f"[Agent 4] Analysis complete. {len(comparisons)} pairs analyzed.")
    print(f"  → Agreement: {stats['agreement']} | Contradiction: {stats['contradiction']} "
          f"| Partial: {stats['partial']} | Novel: {stats['novel']}")

    return {
        "agent": "Agent_4_Agreement_Disagreement",
        "claims_by_paper": {pid: [c["text"] for c in claims] for pid, claims in all_claims.items()},
        "all_claims_structured": all_claims,
        "comparisons": comparisons,
        "summary": summary
    }


if __name__ == "__main__":
    # Quick smoke test
    sample_papers = {
        "paper_A": (
            "We demonstrate that transformer models achieve state-of-the-art results on NLP benchmarks. "
            "Our model outperforms previous baselines by 12% on GLUE. "
            "We show that attention mechanisms are crucial for long-range dependencies. "
            "Results indicate that pre-training on large corpora significantly improves downstream performance."
        ),
        "paper_B": (
            "We find that convolutional architectures can match transformer performance on text classification. "
            "Our experiments show that attention is not always necessary for effective NLP. "
            "The proposed CNN model achieves 89% accuracy, comparable to transformer baselines. "
            "We demonstrate that simpler architectures reduce computational cost without sacrificing accuracy."
        )
    }
    result = run_agent_4(sample_papers)
    print(json.dumps(result["summary"], indent=2))
