"""
Agent 5: Uncertainty Propagation & Research Gap Detection

Multi-signal gap score:
    gap_score(node) = (
        low_degree(node)       * 0.30   # poorly connected
        + high_uncertainty      * 0.30   # uncertain claim
        + low_evidence          * 0.20   # weak evidence
        + low_centrality        * 0.20   # not a bridge
    )
    gaps = sorted(nodes, key=gap_score, reverse=True)

Uncertainty propagation through EDG:
    for node in G:
        neighbours = G.neighbors(node)
        u[node] = 0.8 * u[node] + 0.2 * avg(u[neighbours])
"""
import json
import logging
from typing import List, Tuple, Dict

import networkx as nx

from src.models.schemas import Claim, Agreement, ResearchGap, GapType, RelationType
from src.llm_client import call_llm
from src.hallucination_guard import filter_hallucinated_gaps
from src.graph.edg import EpistemicDependencyGraph
from config import UNCERTAINTY_CONFLICT_WEIGHT, UNCERTAINTY_EVIDENCE_WEIGHT, UNCERTAINTY_STABILITY_WEIGHT

logger = logging.getLogger(__name__)

EVIDENCE_STRENGTH_MAP = {"high": 0.1, "medium": 0.4, "low": 0.8}

_GAP_LABEL_SYSTEM = (
    "Write ONE sentence (max 20 words) naming the research gap for this claim "
    "based on its graph analysis scores. "
    "Return ONLY JSON: {\"gap\":\"...\",\"type\":\"theoretical|empirical|methodological\","
    "\"priority\":\"high|medium|low\"}"
)


# ── Uncertainty computation ───────────────────────────────────────────────────

def compute_uncertainty(claim: Claim, agreements: List[Agreement]) -> float:
    ag = [a for a in agreements if a.claim_i_id == claim.id or a.claim_j_id == claim.id]
    conflict_score = (sum(1 for a in ag if a.relation == RelationType.CONTRADICT)
                      / max(len(ag), 1)) if ag else 0.0
    evidence_u  = EVIDENCE_STRENGTH_MAP.get(claim.evidence_strength, 0.5)
    if claim.assumptions:
        from src.models.schemas import VerificationStatus
        weak = sum(1 for a in claim.assumptions if a.verification == VerificationStatus.WEAK)
        stability_u = weak / len(claim.assumptions)
    else:
        stability_u = 0.5
    return round(min(
        UNCERTAINTY_CONFLICT_WEIGHT * conflict_score +
        UNCERTAINTY_EVIDENCE_WEIGHT * evidence_u +
        UNCERTAINTY_STABILITY_WEIGHT * stability_u,
        1.0
    ), 3)


def propagate_uncertainty(claims: List[Claim], agreements: List[Agreement]) -> List[Claim]:
    """
    Step 1: compute local uncertainty per claim.
    Step 2: propagate through EDG — uncertainty bleeds 20% to neighbours.
    """
    for c in claims:
        c.uncertainty = compute_uncertainty(c, agreements)

    # Build a quick undirected graph for propagation
    G = nx.Graph()
    for c in claims:
        G.add_node(c.id, uncertainty=c.uncertainty)
    for a in agreements:
        if a.relation != RelationType.UNRELATED:
            G.add_edge(a.claim_i_id, a.claim_j_id)

    claim_map = {c.id: c for c in claims}
    current_u = {c.id: c.uncertainty for c in claims}

    for nid in G.nodes:
        if nid not in claim_map:
            continue
        neighbours = [v for v in G.neighbors(nid) if v in claim_map]
        if neighbours:
            avg_n = sum(current_u[v] for v in neighbours) / len(neighbours)
            claim_map[nid].uncertainty = round(
                min(current_u[nid] * 0.8 + avg_n * 0.2, 1.0), 3
            )

    logger.info("Agent 5: Uncertainty propagated for %d claims.", len(claims))
    return claims


# ── Multi-signal gap score ────────────────────────────────────────────────────

def gap_score(
    node_id: str,
    claim: Claim,
    G: nx.Graph,
    bc: Dict[str, float],
) -> float:
    """
    Multi-signal gap score:
        gap_score = low_degree * 0.30
                  + high_uncertainty * 0.30
                  + low_evidence * 0.20
                  + low_centrality * 0.20
    """
    low_degree       = 1.0 / max(G.degree(node_id), 1)
    high_uncertainty = claim.uncertainty
    low_evidence     = EVIDENCE_STRENGTH_MAP.get(claim.evidence_strength, 0.5)
    low_centrality   = 1.0 - bc.get(node_id, 0.0)

    return round(
        low_degree * 0.30 + high_uncertainty * 0.30 +
        low_evidence * 0.20 + low_centrality * 0.20,
        3
    )


def _label_gap(claim: Claim, score: float, bc_val: float, deg: int) -> dict:
    """LLM labels a graph-identified gap. ~35 tokens."""
    payload = json.dumps({
        "claim": f"{claim.subject} {claim.predicate} {claim.object}"[:65],
        "domain": claim.domain[:20],
        "gap_score": score,
        "degree": deg,
        "betweenness": round(bc_val, 3),
        "uncertainty": claim.uncertainty,
    }, separators=(",", ":"))
    res = call_llm(payload, system=_GAP_LABEL_SYSTEM, max_tokens=80)
    if res and isinstance(res, dict) and res.get("gap", "").strip():
        return res
    return {
        "gap": f"Under-studied: {claim.subject} {claim.predicate} {claim.object}"[:70],
        "type": "empirical",
        "priority": "high" if score > 0.6 else "medium",
    }


# ── Gap detection ─────────────────────────────────────────────────────────────

def detect_gaps(
    claims: List[Claim],
    edg: EpistemicDependencyGraph,
) -> Tuple[List[ResearchGap], int]:
    """
    Rank all claims by multi-signal gap score.
    Top-ranked are gaps. LLM labels them.
    """
    if not claims:
        return [], 0

    G   = edg.G.to_undirected() if edg.G else nx.Graph()
    bc  = nx.betweenness_centrality(G, normalized=True)
    claim_map = {c.id: c for c in claims}

    # Score every claim
    scored = []
    for c in claims:
        if c.id not in G.nodes:
            G.add_node(c.id)
        sc = gap_score(c.id, c, G, bc)
        scored.append((c.id, sc))

    # Sort descending — highest gap score first
    scored.sort(key=lambda x: x[1], reverse=True)

    # Take top N (at least 2, at most 6)
    n_gaps = max(2, min(6, len(scored) // 2 + 1))
    top_gap_ids = [nid for nid, _ in scored[:n_gaps]]

    logger.info(
        "Agent 5: gap scores = %s",
        {nid: sc for nid, sc in scored},
    )

    raw_gaps: List[ResearchGap] = []
    for nid in top_gap_ids:
        claim = claim_map.get(nid)
        if not claim:
            continue
        sc  = dict(scored).get(nid, 0)
        bc_val = bc.get(nid, 0.0)
        deg    = G.degree(nid)
        labeled = _label_gap(claim, sc, bc_val, deg)
        g = ResearchGap(
            gap=labeled.get("gap", ""),
            type=labeled.get("type", GapType.EMPIRICAL),
            priority=labeled.get("priority", "medium"),
            related_claims=[nid],
            uncertainty_score=claim.uncertainty,
            gap_signals={
                "degree":      deg,
                "betweenness": round(bc_val, 3),
                "uncertainty": claim.uncertainty,
                "evidence":    claim.evidence_strength or "unknown",
                "gap_score":   sc,
                "signals_fired": [
                    s for s, fired in [
                        ("low connectivity",  deg < 2),
                        ("high uncertainty",  claim.uncertainty > 0.4),
                        ("weak evidence",     claim.evidence_strength in ("low","medium",None,"")),
                        ("low centrality",    bc_val < 0.05),
                    ] if fired
                ],
            },
        )
        if g.gap.strip():
            raw_gaps.append(g)

    # [V4] grounding check
    source_claims = [claim_map[n] for n in top_gap_ids if n in claim_map]
    grounded_gaps, dropped = filter_hallucinated_gaps(raw_gaps, source_claims)
    logger.info("Agent 5: %d gaps — %d dropped by V4.", len(grounded_gaps), dropped)
    return grounded_gaps, dropped
