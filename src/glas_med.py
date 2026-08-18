"""Clinical evidence primitives used by the GLAS-Med pipeline.

The functions in this module implement the paper's deterministic parts: Oxford
evidence-tier assignment, eight-factor reliability scoring, PICO-clustered
reliability-weighted agreement, and uncertainty-impact prioritisation.  LLMs
may extract claims, but these calculations deliberately remain reproducible.
"""
from __future__ import annotations

from collections import defaultdict
from math import log
import re
from typing import Any, Iterable

from src.models.schemas import Agreement, Claim, RelationType, ResearchGap


_WORD_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {"with", "from", "that", "this", "these", "those", "study", "patients", "patient", "using", "were", "was", "and", "the", "for", "into"}


def _text_tokens(value: str) -> set[str]:
    return {token for token in _WORD_RE.findall((value or "").lower()) if len(token) > 2 and token not in _STOPWORDS}


def evidence_tier(text: str, metadata: dict[str, Any] | None = None) -> int:
    """Return the Oxford-style evidence tier (1 is strongest)."""
    haystack = f"{text or ''} {(metadata or {}).get('study_design', '')} {(metadata or {}).get('title', '')}".lower()
    if re.search(r"systematic review|meta[- ]analysis|\bcochrane\b", haystack):
        return 1
    if re.search(r"randomi[sz]ed|\brct\b|controlled trial|clinical trial|\bphase [1-4]\b|double[- ]blind|placebo[- ]controlled", haystack):
        return 2
    if re.search(r"prospective cohort|retrospective cohort|\bcohort\b|\blongitudinal\b|\bobservational\b|real[- ]world|subgroup analys|registry study", haystack):
        return 3
    if re.search(r"case[- ]control|cross[- ]sectional|case series", haystack):
        return 4
    return 5


def _sample_size(text: str, metadata: dict[str, Any]) -> int:
    supplied = metadata.get("sample_size") or metadata.get("dataset_size") or metadata.get("n")
    if isinstance(supplied, (int, float)):
        return max(0, int(supplied))
    match = re.search(r"\b(?:n\s*=\s*|enrolled\s+|included\s+)([0-9][0-9,]*)", text or "", re.I)
    return int(match.group(1).replace(",", "")) if match else 0


def study_reliability(text: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    """Compute the paper's eight-factor reliability score rho in [0, 1]."""
    metadata = metadata or {}
    haystack = (text or "").lower()
    tier = evidence_tier(text, metadata)
    design_weights = {1: 1.0, 2: 1.0, 3: 0.75, 4: 0.55, 5: 0.30}
    factors = {"design": design_weights[tier]}
    n = _sample_size(text, metadata)
    factors["sample_size"] = -0.15 if 0 < n < 50 else 0.0
    factors["blinding_randomisation"] = 0.12 if re.search(r"double[- ]blind|blinded|randomi[sz]ed", haystack) else 0.0
    factors["follow_up"] = -0.10 if re.search(r"loss to follow[- ]up[^.]{0,30}(?:2[1-9]|[3-9][0-9])%|attrition[^.]{0,30}(?:2[1-9]|[3-9][0-9])%", haystack) else 0.0
    factors["statistical_rigour"] = 0.10 if re.search(r"confidence interval|p\s*[<=>]|adjusted (?:odds|hazard)|pre-?registered", haystack) else 0.0
    factors["industry_sponsorship"] = -0.08 if re.search(r"funded by|sponsored by", haystack) else 0.0
    factors["preregistration"] = 0.08 if re.search(r"clinicaltrials\.gov|\bnct\d{5,}\b|pre-?registered", haystack) else 0.0
    quartile = metadata.get("journal_impact_quartile")
    factors["journal_impact"] = {1: 0.0, 2: -0.05, 3: -0.10, 4: -0.15}.get(quartile, 0.0)
    rho = round(max(0.0, min(1.0, sum(factors.values()))), 3)
    return {"score": rho, "tier": tier, "sample_size": n, "quarantined": rho < 0.45, "factors": factors}


def pico_for_claim(claim: Claim) -> dict[str, str]:
    """Create an auditable lightweight PICO projection from extracted claims."""
    subject = (claim.subject or "").strip()
    outcome = (claim.object or "").strip()
    method = (claim.method or "").strip()
    return {
        "population": claim.domain or "unspecified population",
        "intervention": subject,
        "comparator": method if re.search(r"versus|vs\.?|compar", method, re.I) else "unspecified comparator",
        "outcome": outcome,
    }


def attach_provenance(claims: Iterable[Claim], paper: dict[str, Any]) -> list[Claim]:
    full_text = paper.get("text") or paper.get("abstract") or paper.get("summary") or ""
    if paper.get("title") and paper.get("title") not in full_text:
        full_text = f"{paper.get('title')}. {full_text}"
    report = study_reliability(full_text, paper)
    for claim in claims:
        claim.extraction_confidence = max(0.0, min(1.0, float(getattr(claim, "extraction_confidence", 0.5) or 0.5)))
        claim.evidence_tier = report["tier"]
        claim.study_reliability = report["score"]
        claim.pico = pico_for_claim(claim)
        claim.provenance = {
            "paper_id": paper.get("id", claim.paper_id),
            "pmid": paper.get("pmid", ""),
            "trial_id": paper.get("trial_id", ""),
            "sample_size": report["sample_size"],
            "design_tier": report["tier"],
            "citation_count": int(paper.get("citation_count") or 0),
            "reliability_factors": report["factors"],
            "quarantined": report["quarantined"],
        }
    return list(claims)


def _cluster_key(claim: Claim) -> str:
    pico = claim.pico or pico_for_claim(claim)
    intervention = " ".join(sorted(_text_tokens(pico.get("intervention", ""))))
    outcome = " ".join(sorted(_text_tokens(pico.get("outcome", ""))))
    return f"{intervention}|{outcome}" or claim.id


def _pico_overlap(left: Claim, right: Claim) -> bool:
    """Match semantically similar PICO records despite different result phrasing."""
    left_pico, right_pico = left.pico or pico_for_claim(left), right.pico or pico_for_claim(right)
    intervention_left = _text_tokens(left_pico.get("intervention", ""))
    intervention_right = _text_tokens(right_pico.get("intervention", ""))
    outcome_left = _text_tokens(left_pico.get("outcome", ""))
    outcome_right = _text_tokens(right_pico.get("outcome", ""))
    if not intervention_left or not intervention_right or not outcome_left or not outcome_right:
        return False
    shared_intervention = intervention_left & intervention_right
    shared_outcome = outcome_left & outcome_right
    if shared_intervention and shared_outcome:
        return True
    intervention_overlap = len(shared_intervention) / len(intervention_left | intervention_right)
    outcome_overlap = len(shared_outcome) / len(outcome_left | outcome_right)
    return (intervention_overlap >= 0.20 and outcome_overlap >= 0.15) or bool(shared_intervention and outcome_overlap >= 0.10)


def _direction(claim: Claim) -> str:
    text = f"{claim.predicate} {claim.object}".lower()
    if re.search(r"no (?:benefit|effect|difference)|fail|worse|harm|increase[sd]?(?:[_\s]+)risk|adverse", text):
        return "negative"
    if re.search(r"reduce|decrease|improve|benefit|effective|protect|increase[sd]? survival", text):
        return "positive"
    return "neutral"


def weighted_agreements(claims: Iterable[Claim]) -> list[Agreement]:
    """Implement A_k = dominant(sum(rho_i*sigma_i)) / sum(rho_j*sigma_j)."""
    clusters: dict[str, list[Claim]] = defaultdict(list)
    for claim in claims:
        if not claim.provenance.get("quarantined", False):
            # Outcomes often contain different confidence intervals or effect
            # sizes. Cluster by PICO token overlap rather than exact strings.
            matching_key = next(
                (key for key, members in clusters.items() if members and _pico_overlap(claim, members[0])),
                None,
            )
            clusters[matching_key or _cluster_key(claim)].append(claim)

    agreements: list[Agreement] = []
    for cluster, members in clusters.items():
        if len(members) < 2:
            continue
        weights: dict[str, float] = defaultdict(float)
        for claim in members:
            weights[_direction(claim)] += claim.study_reliability * claim.extraction_confidence
        total = sum(weights.values())
        dominant, dominant_weight = max(weights.items(), key=lambda item: item[1])
        score = round(dominant_weight / total, 3) if total else 0.0
        verdict = "Agree" if score > 0.70 else "Partial Agreement" if score >= 0.45 else "Disagree"
        for index, left in enumerate(members):
            for right in members[index + 1:]:
                same_direction = _direction(left) == _direction(right)
                relation = RelationType.AGREE if same_direction else RelationType.CONTRADICT
                agreements.append(Agreement(
                    claim_i_id=left.id, claim_j_id=right.id, relation=relation,
                    confidence=score, reason=f"PICO reliability-weighted {verdict.lower()}",
                    agreement_basis="pico-reliability-weighted", pico_cluster=cluster,
                    weighted_agreement=score, verdict=verdict,
                ))
    return agreements


def uncertainty_priorities(claims: Iterable[Claim], agreements: Iterable[Agreement]) -> list[ResearchGap]:
    """Rank disputed PICO clusters with U_k=(1-A_k)*mean(rho)*log(1+mean(citations))."""
    claim_by_id = {claim.id: claim for claim in claims}
    by_cluster: dict[str, list[Agreement]] = defaultdict(list)
    for agreement in agreements:
        if agreement.pico_cluster:
            by_cluster[agreement.pico_cluster].append(agreement)
    gaps: list[ResearchGap] = []
    for cluster, rows in by_cluster.items():
        ids = {row.claim_i_id for row in rows} | {row.claim_j_id for row in rows}
        members = [claim_by_id[claim_id] for claim_id in ids if claim_id in claim_by_id]
        if not members:
            continue
        agreement_score = rows[0].weighted_agreement
        mean_rho = sum(c.study_reliability for c in members) / len(members)
        mean_citations = sum(float(c.provenance.get("citation_count", 0)) for c in members) / len(members)
        impact = round((1 - agreement_score) * mean_rho * log(1 + mean_citations), 3)
        for claim in members:
            claim.uncertainty = round(1 - agreement_score, 3)
        if rows[0].verdict == "Agree":
            continue
        gaps.append(ResearchGap(
            gap=f"Resolve {rows[0].verdict.lower()} clinical evidence for {cluster.replace('|', ' and ')}.",
            type="empirical", priority="high" if impact >= 0.5 else "medium",
            related_claims=sorted(ids), uncertainty_score=round(1 - agreement_score, 3),
            gap_signals={"uncertainty_impact": impact, "agreement": agreement_score, "mean_reliability": round(mean_rho, 3), "mean_citations": round(mean_citations, 2), "formula": "(1-A_k)*mean(rho_k)*log(1+mean(citations_k))"},
        ))
    return sorted(gaps, key=lambda gap: gap.gap_signals["uncertainty_impact"], reverse=True)
