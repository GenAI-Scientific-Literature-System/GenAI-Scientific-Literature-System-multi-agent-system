"""
agents/agreement_detection/agreement_detection_agent.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Agent 4: Agreement & Contradiction Detection

Derived from GenAI Agent 4 (src/agents/agent4_agreement.py)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
IDENTITY
  agent_id        : "agent_4_agreement_detection"
  genai_origin    : src/agents/agent4_agreement.py

GENAI FORMAL MODEL (preserved exactly from GenAI Agent 4)
  R(Ci, Cj | Ai, Aj) — formal set-operation agreement function:

    def agreement(C1, C2):
        A1 = set(C1["assumptions"])
        A2 = set(C2["assumptions"])
        if   A1 == A2:            return AGREE,        1.0,   "identical-sets"
        elif A1.isdisjoint(A2):   return CONTRADICT,   0.85,  "disjoint-sets"
        else:                     return CONDITIONAL,  jaccard, "partial-overlap"

  Three-tier decision hierarchy (GenAI original):
    Tier 1: Formal assumption set operations    (0 tokens, always first)
    Tier 2: EDG path inference (nx.shortest_path, transitive relations)
    Tier 3: Predicate heuristic                (opposing predicates → CONTRADICT)
    Tier 4: LLM narrates reason only           (~30 tokens, NEVER decides)

ANTI-HALLUCINATION
  [V3] verify_agreement_reason() — checks reason text tokens reference actual
  claim content. Imported from genai_system.src.hallucination_guard.

UNIFIED ARCHITECTURE MODULES USED (wrapped inside this agent)
  - src/reasoning.py     → EDG path inference
  - src/graph/edg.py     → EpistemicDependencyGraph
  - src/struct.py        → MERLINStruct (assumption ID registry)

PIPELINE CONTRACT
  Input  : "agent_3_normalisation" → "claims" (normalised, assumption-enriched)
  Output :
    {
      "agreements": List[Agreement dict],
      "summary": {
        "total_pairs": int, "agree": int, "contradict": int,
        "conditional": int, "unrelated": int
      }
    }
"""
from __future__ import annotations

import logging
from collections import Counter
from typing import Any, Dict, List

from agents.base.base_agent import AgentContext, AgentResult, BaseAgent
from core.registry.agent_registry import registry

logger = logging.getLogger("agent.agent_4_agreement_detection")

# ── Prompt: derived from GenAI Agent 4's _REASON_SYSTEM (preserved) ───────────
# Original system prompt: "Write ONE sentence (max 15 words) explaining the
# epistemic relation between two scientific claims given their assumption IDs."
# Expanded here into a full prompt_template (Unified architecture pattern).
AGREEMENT_REASON_PROMPT = """\
You are Agent 4 — a scientific agreement reasoning narrator.
Derived from GenAI Agent 4 (src/agents/agent4_agreement.py).

IMPORTANT: The formal relation has ALREADY been determined by set-operation logic.
Your ONLY role: write ONE concise sentence (≤ 20 words) explaining WHY
claims Ci and Cj have this relationship, given their assumption overlap.

Ci predicate : {ci_pred}
Cj predicate : {cj_pred}
Shared assumptions: {shared}
Relation decided  : {relation}
Basis             : {basis}

Return ONLY: {{"reason": "<your one-sentence explanation>"}}
"""


@registry.register
class AgreementDetectionAgent(BaseAgent):
    """
    Agent 4: Detects pairwise claim agreement using formal set-operation reasoning.

    Derived from GenAI Agent 4 (src/agents/agent4_agreement.py)
    Wraps: genai_system.src.agents.agent4_agreement.compute_agreements()

    The full four-tier decision hierarchy from the original GenAI codebase is
    preserved here — this class adds only the BaseAgent interface contract.

    FORMAL LOGIC (from GenAI, unchanged):
      1. agreement() — assumption set-ops decide relation (0 LLM tokens)
      2. _infer_via_path() — EDG shortest-path for transitive inference
      3. _predicate_heuristic() — opposing predicates for unknown pairs
      4. call_mistral() — writes readable reason AFTER decision (~30 tokens)
    """

    agent_id        = "agent_4_agreement_detection"
    role            = (
        "Computes pairwise claim agreement using formal assumption set logic: "
        "R(Ci, Cj | Ai, Aj). "
        "AGREE if A1==A2; CONTRADICT if A1∩A2=∅ with predicate conflict; "
        "CONDITIONAL on partial overlap (Jaccard confidence). "
        "EDG path inference for transitive relations. "
        "LLM writes only a short reason post-decision (~30 tokens). "
        "Applies V3 Anti-Hallucination: verify_agreement_reason()."
    )
    prompt_template = AGREEMENT_REASON_PROMPT
    genai_origin    = "src/agents/agent4_agreement.py"   # GenAI credit

    required_context_keys: List[str] = [
        "agent_3_normalisation.claims",
    ]

    def _execute(self, context: AgentContext) -> AgentResult:
        """
        Delegates to genai_system.src.agents.agent4_agreement.compute_agreements().

        That function implements the complete GenAI formal agreement pipeline:
          Step 1  agreement(ci, cj, struct) — assumption set operations
          Step 2  _infer_via_path()          — EDG transitive inference
          Step 3  _predicate_heuristic()     — predicate opposition check
          Step 4  call_mistral()             — reason narration only (~30 tokens)
          Step V3 verify_agreement_reason()  — hallucination guard
        """
        try:
            from genai_system.src.agents.agent4_agreement import compute_agreements
            from genai_system.src.models.schemas import Claim
            from genai_system.src.struct import MERLINStruct
        except ImportError as exc:
            return self._fail(
                f"[Derived from GenAI Agent 4] Could not import compute_agreements: {exc}"
            )

        raw_claims: List[Dict[str, Any]] = context.upstream(
            "agent_3_normalisation", "claims", default=[]
        )
        if len(raw_claims) < 2:
            logger.warning(
                "[agent_4] Need ≥2 claims for agreement detection; got %d.", len(raw_claims)
            )
            return self._ok({
                "agreements": [],
                "summary": {"total_pairs": 0, "agree": 0, "contradict": 0,
                            "conditional": 0, "unrelated": 0},
            })

        # Reconstruct typed Claim objects from serialised dicts
        try:
            claims_objs = [
                Claim(**{k: v for k, v in c.items() if k in Claim.__dataclass_fields__})
                for c in raw_claims
            ]
        except Exception as exc:
            return self._fail(f"Claim reconstruction failed: {exc}")

        # Build MERLINStruct (assumption ID registry) from assumption-enriched claims.
        # GenAI Agent 4 uses struct.claims[id]["assumptions"] for set-ops.
        struct = MERLINStruct()
        for claim in claims_objs:
            assumption_ids = [
                getattr(a, "id", str(i))
                for i, a in enumerate(getattr(claim, "assumptions", []))
            ]
            struct.claims[claim.id] = {
                "subj":        claim.subject,
                "pred":        claim.predicate,
                "obj":         claim.object,
                "domain":      claim.domain,
                "assumptions": assumption_ids,
            }
            # Register assumptions in struct so shared_ids resolve to constraint text
            for i, a in enumerate(getattr(claim, "assumptions", [])):
                aid = getattr(a, "id", str(i))
                struct.assumptions[aid] = {
                    "constraint": getattr(a, "constraint", str(a))[:60]
                }

        # ── Delegate to GenAI Agent 4 ─────────────────────────────────────────
        # compute_agreements() runs the full four-tier decision hierarchy and
        # builds an incremental NetworkX DiGraph for EDG path inference.
        try:
            from genai_system.src.models.schemas import Agreement
            agreements: List[Agreement] = compute_agreements(claims_objs, struct)
        except Exception as exc:
            logger.error("[agent_4] compute_agreements raised: %s", exc)
            return self._fail(f"Agreement computation failed: {exc}")

        serialised = [
            a.to_dict() if hasattr(a, "to_dict") else vars(a)
            for a in agreements
        ]

        # Summary statistics
        rel_counts = Counter(
            getattr(a, "relation", a.get("relation", "unrelated"))
            if not isinstance(a, dict) else a.get("relation", "unrelated")
            for a in agreements
        )
        summary = {
            "total_pairs": len(serialised),
            "agree":       rel_counts.get("agree",       0),
            "contradict":  rel_counts.get("contradict",  0),
            "conditional": rel_counts.get("conditional", 0),
            "unrelated":   rel_counts.get("unrelated",   0),
        }

        logger.info(
            "[agent_4] %d pairs | AGREE:%d CONTRADICT:%d CONDITIONAL:%d UNRELATED:%d",
            summary["total_pairs"], summary["agree"], summary["contradict"],
            summary["conditional"], summary["unrelated"],
        )

        return self._ok(
            payload={"agreements": serialised, "summary": summary},
            pair_count=summary["total_pairs"],
            contradiction_count=summary["contradict"],
            basis_used="formal-set-ops+EDG-path+predicate-heuristic",
        )
