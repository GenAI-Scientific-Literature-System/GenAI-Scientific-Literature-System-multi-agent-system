"""
MERLIN Pipeline — Structure-First, Assumption-Consistent Architecture
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PHASE 1 — EXTRACTION  (text used here and ONLY here)
  text → RAG retriever → Agent 1 [V1] → Agent 2 [V2] → Agent 3
       → Agent 6 → Agent 6.1 [V5]
       → AssumptionEngine.validate_all()  ← actively rejects bad claims
       → MERLINStruct.build()             ← TEXT DISCARDED

PHASE 2 — REASONING  (MERLINStruct + Claim objects only, zero raw text)
  struct → Agent 4: set-op decides relation, LLM writes reason only
         → Agent 5: uncertainty propagation
         → EDG: betweenness + pagerank + clustering + shortest path
         → gap_regions(degree<2 OR bc<0.05) → LLM labels gaps
         → formal epistemic loss score
"""
import logging
import time
from typing import List, Dict, Any

from src.agents.agent1_claim       import extract_claims
from src.agents.agent2_evidence    import attribute_evidence
from src.agents.agent3_normalize   import normalise_claims
from src.agents.agent6_assumption  import extract_assumptions, assign_assumptions_to_claims
from src.agents.agent6_1_verify    import verify_all_assumptions
from src.agents.agent4_agreement   import compute_agreements
from src.agents.agent5_uncertainty import propagate_uncertainty, detect_gaps
from src.graph.edg                 import build_edg, EpistemicDependencyGraph
from src.hallucination_guard       import HallucinationReport
from src.models.schemas            import Claim, Agreement, ResearchGap
from src.document_store            import get_struct, put_struct
from src.retrieval                 import DocumentRetriever
from src.struct                    import MERLINStruct
from src.reasoning                 import formal_score
from src.assumption_engine         import validate_all as ace_validate
from src.mistral_client            import get_token_usage, reset_token_log

logger = logging.getLogger(__name__)


class MERLINResult:
    def __init__(self):
        self.claims:              List[Claim]              = []
        self.agreements:          List[Agreement]          = []
        self.gaps:                List[ResearchGap]        = []
        self.edg:                 EpistemicDependencyGraph = None
        self.elapsed_sec:         float                    = 0.0
        self.token_stats:         Dict[str, Any]           = {}
        self.ace_report:          Dict[str, Any]           = {}
        self.hallucination_report: HallucinationReport     = HallucinationReport()

    def to_dict(self) -> Dict[str, Any]:
        contradictions = sum(1 for a in self.agreements if a.relation == "contradict")
        total_pairs    = len(self.agreements)
        avg_u          = sum(c.uncertainty for c in self.claims) / max(len(self.claims), 1)
        total_rejected = self.hallucination_report.v5_assumptions_rejected
        total_assump   = total_rejected + sum(len(c.assumptions) for c in self.claims)
        rejection_rate = total_rejected / max(total_assump, 1)

        epistemic_loss = formal_score(
            contradiction_count=contradictions,
            total_pairs=total_pairs,
            avg_uncertainty=round(avg_u, 3),
            assumption_rejection_rate=rejection_rate,
        )

        return {
            "claims":     [c.to_dict() for c in self.claims],
            "agreements": [a.to_dict() for a in self.agreements],
            "gaps":       [g.to_dict() for g in self.gaps],
            "graph":      self.edg.to_dict() if self.edg else {},
            "hallucination_report": self.hallucination_report.to_dict(),
            "ace_report":           self.ace_report,
            "token_stats":          self.token_stats,
            "meta": {
                "elapsed_sec":          round(self.elapsed_sec, 2),
                "total_claims":         len(self.claims),
                "total_gaps":           len(self.gaps),
                "contradictions":       contradictions,
                "hallucination_guards": self.hallucination_report.total_interventions,
                "epistemic_loss":       epistemic_loss,
                "avg_uncertainty":      round(avg_u, 3),
                "mistral_tokens":       self.token_stats.get("mistral_total", 0),
                "mistral_calls":        self.token_stats.get("mistral_calls", 0),
                "cache_hits":           self.token_stats.get("cache_hits", 0),
                "ace_rejected":         self.ace_report.get("rejected", 0),
            },
        }


def run_pipeline(papers: List[Dict[str, str]]) -> MERLINResult:
    t0         = time.time()
    result     = MERLINResult()
    hr         = result.hallucination_report
    all_claims: List[Claim] = []
    all_assumptions          = []
    total_chunks = 0
    ace_reports  = []

    reset_token_log()
    logger.info("MERLIN pipeline starting for %d paper(s).", len(papers))

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 1 — EXTRACTION
    # ══════════════════════════════════════════════════════════════════════════
    for paper in papers:
        pid  = paper.get("id", "unknown")
        text = paper.get("text", "")
        if not text.strip():
            continue

        retriever     = DocumentRetriever(text, paper_id=pid)
        total_chunks += retriever.chunk_count

        cached = get_struct(text)
        if cached:
            logger.info("Pipeline: cache hit '%s'.", pid)
            import uuid
            from src.models.schemas import Claim as C
            claims = [C(
                id=d.get("id", str(uuid.uuid4())[:8]),
                subject=d.get("subject",""), predicate=d.get("predicate",""),
                object=d.get("object",""), method=d.get("method",""),
                domain=d.get("domain",""), paper_id=pid,
            ) for d in cached["claims"]]
            all_claims.extend(claims)
            continue

        # Agent 1 [V1]
        claims, v1 = extract_claims(text, paper_id=pid, retriever=retriever)
        hr.v1_claims_dropped += v1

        # Agent 2 [V2]
        claims, v2 = attribute_evidence(claims, text, retriever=retriever)
        hr.v2_spans_removed += v2

        # Agent 3
        claims = normalise_claims(claims)

        # Agent 6
        assumptions = extract_assumptions(text, claims, retriever=retriever)

        # Agent 6.1 [V5]
        before_a             = len(assumptions)
        verified_assumptions = verify_all_assumptions(assumptions, text)
        hr.v5_assumptions_rejected += max(0, before_a - len(verified_assumptions))

        claims = assign_assumptions_to_claims(claims, verified_assumptions)

        # ── Assumption-Consistency Engine (active rejection) ──────────────────
        valid_claims, invalid_claims, ace_rpt = ace_validate(claims)
        ace_reports.append(ace_rpt)
        if invalid_claims:
            logger.info("ACE: rejected %d claims for '%s'.", len(invalid_claims), pid)
        claims = valid_claims  # only valid claims proceed

        put_struct(text, pid,
                   claims=[c.to_dict() for c in claims],
                   assumptions=[a.to_dict() for a in verified_assumptions],
                   chunk_count=retriever.chunk_count)

        all_assumptions.extend(verified_assumptions)
        all_claims.extend(claims)

    result.claims = all_claims

    # Aggregate ACE report
    result.ace_report = {
        "validated": sum(r.get("validated", 0) for r in ace_reports),
        "rejected":  sum(r.get("rejected",  0) for r in ace_reports),
        "ungrounded":sum(r.get("ungrounded",0) for r in ace_reports),
    }

    if not all_claims:
        logger.warning("No claims after extraction + ACE.")
        result.elapsed_sec = time.time() - t0
        return result

    # ── Build MERLINStruct — TEXT DISCARDED ───────────────────────────────────
    struct = MERLINStruct.build(all_claims, all_assumptions)

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 2 — REASONING  (struct + Claim objects only)
    # ══════════════════════════════════════════════════════════════════════════

    # Agent 4: set-op decides relation, LLM only writes reason
    agreements        = compute_agreements(all_claims, struct)
    result.agreements = agreements
    hr.v3_reasons_rewritten = sum(
        1 for a in agreements if a.reason and a.reason.startswith("[Auto-summary]")
    )

    # Agent 5: uncertainty propagation
    all_claims    = propagate_uncertainty(all_claims, agreements)
    result.claims = all_claims

    # Update uncertainty in struct
    for c in all_claims:
        if c.id in struct.claims:
            struct.claims[c.id]["uncertainty"] = c.uncertainty

    # EDG: build with full analytics
    edg        = build_edg(all_claims, agreements)
    result.edg = edg

    # Gap detection: graph-first, LLM labels only
    raw_gaps, v4_dropped = detect_gaps(all_claims, edg)
    hr.v4_gaps_dropped   = v4_dropped
    result.gaps          = raw_gaps

    # Token accounting
    tok = get_token_usage()
    result.token_stats = {
        "chunks_indexed":     total_chunks,
        "mistral_prompt":     tok["prompt_tokens"],
        "mistral_completion": tok["completion_tokens"],
        "mistral_total":      tok["total_tokens"],
        "mistral_calls":      tok["api_calls"],
        "cache_hits":         tok["cache_hits"],
        "text_sent_to_llm":   "NEVER after extraction phase",
    }

    result.elapsed_sec = round(time.time() - t0, 2)
    logger.info(
        "MERLIN complete %.1fs | claims=%d agreements=%d gaps=%d | "
        "tokens=%d calls=%d | ACE rejected=%d",
        result.elapsed_sec, len(all_claims), len(agreements), len(raw_gaps),
        tok["total_tokens"], tok["api_calls"], result.ace_report.get("rejected", 0),
    )
    return result
