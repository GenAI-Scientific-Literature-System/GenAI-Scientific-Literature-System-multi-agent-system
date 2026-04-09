"""
Agent 4: Agreement Reasoning  R(Ci, Cj | Ai, Aj)

THE FORMAL AGREEMENT FUNCTION:

    claims = {
        "C1": {"text": "...", "assumptions": ["A1","A2"]},
        "C2": {"text": "...", "assumptions": ["A2","A3"]}
    }

    def agreement(C1, C2):
        A1 = set(C1["assumptions"])
        A2 = set(C2["assumptions"])
        if   A1 == A2:            return "AGREE"
        elif A1.isdisjoint(A2):   return "CONTRADICTION"
        else:                     return "CONDITIONAL"

This is the COMPLETE decision logic. LLM writes a readable reason
string AFTER the decision is already made (~30 tokens max).

EDG path inference additionally checks for TRANSITIVE relations
between non-adjacent claims via shortest path analysis.
"""
import json
import logging
from itertools import combinations
from typing import List, Tuple, Optional

import networkx as nx

from src.models.schemas import Claim, Agreement, RelationType
from src.mistral_client import call_mistral
from src.hallucination_guard import verify_agreement_reason
from src.struct import MERLINStruct

logger = logging.getLogger(__name__)


# ── The formal agreement function ────────────────────────────────────────────

def agreement(c1_id: str, c2_id: str, struct: MERLINStruct):
    """
    Formal set-operation agreement.  Zero LLM tokens.
    Returns (relation, confidence, basis, shared_ids).
    """
    A1 = set(struct.claims.get(c1_id, {}).get("assumptions", []))
    A2 = set(struct.claims.get(c2_id, {}).get("assumptions", []))
    shared = list(A1 & A2)

    if not A1 and not A2:
        return "unknown", 0.0, "no-assumptions", []

    if A1 == A2:
        return RelationType.AGREE, 1.0, "identical-sets", shared

    if A1.isdisjoint(A2):
        return RelationType.CONTRADICT, 0.85, "disjoint-sets", []

    jaccard = len(A1 & A2) / len(A1 | A2)
    return RelationType.CONDITIONAL, round(jaccard, 3), "partial-overlap", shared


def _predicate_heuristic(ci_data: dict, cj_data: dict) -> Tuple[str, float]:
    """Fast structural pre-check for claims with no assumptions."""
    pi, pj = ci_data.get("pred", ""), cj_data.get("pred", "")
    di, dj = ci_data.get("domain", ""), cj_data.get("domain", "")
    if di and dj and di != dj:
        return RelationType.UNRELATED, 0.92
    opposing = {("outperforms","underperforms"),("improves","reduces"),("demonstrates","fails")}
    if (pi, pj) in opposing or (pj, pi) in opposing:
        return RelationType.CONTRADICT, 0.85
    if pi == pj:
        return RelationType.AGREE, 0.70
    return "unknown", 0.0


def _infer_via_path(G: nx.DiGraph, ci_id: str, cj_id: str) -> Optional[Tuple[str, float]]:
    """
    EDG path inference for non-adjacent claim pairs.
    If there is a path between C1 and C2 through existing edges,
    analyse the edge types along it to infer a transitive relation.
    """
    try:
        path = nx.shortest_path(G.to_undirected(), ci_id, cj_id)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None

    if len(path) <= 2:
        return None  # direct edge — not path inference

    edges_on_path = []
    for i in range(len(path) - 1):
        d = G.get_edge_data(path[i], path[i+1]) or G.get_edge_data(path[i+1], path[i]) or {}
        edges_on_path.append(d.get("relation", "unknown"))

    contra = sum(1 for r in edges_on_path if r == "contradict")
    agrees = sum(1 for r in edges_on_path if r == "agree")

    if contra > 0:
        return RelationType.CONDITIONAL, 0.55   # indirect contradiction → conditional
    if agrees == len(edges_on_path):
        return RelationType.AGREE, 0.65          # transitive support
    return None


_REASON_SYSTEM = (
    "Write ONE sentence (max 15 words) explaining the epistemic relation "
    "between two scientific claims given their assumption IDs. "
    "Return only the sentence, no JSON, no preamble."
)


def compute_agreements(claims: List[Claim], struct: MERLINStruct) -> List[Agreement]:
    """
    1. agreement(C1,C2) decides relation from assumption ID sets
    2. EDG path inference for unresolved pairs
    3. Predicate heuristic as last structural fallback
    4. LLM writes readable reason only (~30 tokens)
    """
    # Build a partial graph incrementally for path inference
    G = nx.DiGraph()
    for c in claims:
        G.add_node(c.id)

    agreements = []

    for ci, cj in combinations(claims, 2):
        ci_data = struct.claims.get(ci.id, {})
        cj_data = struct.claims.get(cj.id, {})

        # ── Step 1: formal set-operation agreement ────────────────────────────
        relation, confidence, basis, shared_ids = agreement(ci.id, cj.id, struct)
        source = basis

        # ── Step 2: EDG path inference for "unknown" ─────────────────────────
        if relation == "unknown" and G.number_of_edges() > 0:
            path_result = _infer_via_path(G, ci.id, cj.id)
            if path_result:
                relation, confidence = path_result
                source = "path-inference"
                shared_ids = []

        # ── Step 3: predicate heuristic if still unknown ──────────────────────
        if relation == "unknown":
            relation, confidence = _predicate_heuristic(ci_data, cj_data)
            source = "predicate-heuristic"
            shared_ids = []

        if relation == "unknown":
            relation, confidence = RelationType.UNRELATED, 0.5
            source = "default"
            shared_ids = []

        # ── Step 4: LLM writes reason (~30 tokens) — NEVER decides relation ───
        reason = f"{source}:{relation}"
        A1 = list(struct.claims.get(ci.id, {}).get("assumptions", []))
        A2 = list(struct.claims.get(cj.id, {}).get("assumptions", []))
        if A1 or A2:
            prompt = json.dumps({
                "C1_pred": ci_data.get("pred","")[:20],
                "C2_pred": cj_data.get("pred","")[:20],
                "A1": A1[:3], "A2": A2[:3], "relation": relation,
            }, separators=(",",":"))
            res = call_mistral(prompt, system=_REASON_SYSTEM, max_tokens=40)
            if res:
                if isinstance(res, str):
                    raw = res
                elif isinstance(res, list):
                    raw = " ".join(str(x) for x in res)
                elif isinstance(res, dict):
                    raw = res.get("reason") or res.get("text") or ""
                else:
                    raw = str(res)
                if str(raw).strip():
                    reason = str(raw).strip()[:120]

        # Resolve shared assumption IDs → constraint text for display
        shared_texts = [
            struct.assumptions.get(aid, {}).get("constraint", aid)[:40]
            for aid in shared_ids
        ]
        ag = Agreement(
            claim_i_id=ci.id, claim_j_id=cj.id,
            relation=relation, confidence=round(confidence, 3), reason=reason,
            agreement_basis=source,
            shared_assumptions=shared_texts,
        )

        # [V3] grounding check
        ci_text = f"{ci_data.get('subj','')} {ci_data.get('pred','')} {ci_data.get('obj','')}"
        cj_text = f"{cj_data.get('subj','')} {cj_data.get('pred','')} {cj_data.get('obj','')}"
        verify_agreement_reason(ag, ci_text, cj_text)

        # Add edge to partial graph so later pairs can use path inference
        G.add_edge(ci.id, cj.id, relation=relation, confidence=confidence)
        struct.add_relation(ci.id, cj.id, ag.relation, ag.confidence, ag.reason)
        agreements.append(ag)

    logger.info(
        "Agent 4 [formal]: %d pairs — AGREE:%d CONTRADICT:%d CONDITIONAL:%d UNRELATED:%d",
        len(agreements),
        sum(1 for a in agreements if a.relation == RelationType.AGREE),
        sum(1 for a in agreements if a.relation == RelationType.CONTRADICT),
        sum(1 for a in agreements if a.relation == RelationType.CONDITIONAL),
        sum(1 for a in agreements if a.relation == RelationType.UNRELATED),
    )
    return agreements
