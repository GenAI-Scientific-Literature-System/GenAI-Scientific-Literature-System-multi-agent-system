"""
Agent 5: Epistemic Boundary & Research Gap Engine
Identifies uncertainty and research gaps via epistemic stress testing and boundary analysis.
Consumes Agent 4 outputs as input signals.
"""

import json
import os
import uuid
from typing import List, Dict, Any
from mistralai import Mistral

from models import (
    Agent5Output, BoundaryType, TraceInfo,
    HypothesisInput, Agent4Output, ResearchGap, CompatibilityType
)

TRACE = TraceInfo(component="Agent_5", method="boundary_analysis")

SYSTEM_PROMPT = """You are a AAAI-level epistemic reasoner specializing in scientific 
boundary detection and research gap discovery. You identify where scientific knowledge 
breaks down, fails to generalize, or leaves critical questions unanswered.

Your analysis must:
1. Identify epistemic boundaries — regions where hypotheses fail or become unreliable
2. Stress-test hypotheses against distribution shifts and extreme conditions
3. Discover unknown-unknown dimensions not addressed in any input
4. Quantify epistemic risk and information gain potential

You MUST respond with valid JSON only — no preamble, no markdown fences."""

BOUNDARY_PROMPT = """Perform epistemic boundary analysis for this scientific cluster:

HYPOTHESES UNDER ANALYSIS:
{hypotheses_block}

COMPATIBILITY CONTEXT FROM AGENT 4:
{compat_context}

TASK: Identify ONE major epistemic boundary where scientific knowledge breaks down.

Stress-test the hypotheses by simulating:
1. Distribution shifts (different populations, environments, scales)
2. Extreme conditions (edge cases, boundary values)
3. Missing variables (unmeasured confounders, latent factors)

Determine:
- Where do these hypotheses stop generalizing?
- What assumptions break under stress?
- What dimension of inquiry is entirely absent?
- Is this an "unknown-unknown" (a question no paper even asks)?

Respond with this exact JSON:
{{
  "boundary_type": "GENERALIZATION_FAILURE|ASSUMPTION_BREAKDOWN|UNEXPLORED_DIMENSION",
  "failure_conditions": ["list of specific conditions under which hypotheses fail"],
  "stress_test_summary": "2-3 sentence summary of stress test results",
  "unknown_unknown_indicator": true/false,
  "research_gap": {{
    "description": "precise description of the gap",
    "reason_unresolved": "why this has not been resolved",
    "suggested_investigation": "concrete next research step"
  }},
  "epistemic_risk_score": 0.0,
  "information_gain_score": 0.0,
  "counterfactual_probes": ["list of 2-3 probing questions that expose the gap"],
  "confidence_score": 0.0
}}

Scoring:
- epistemic_risk_score: How dangerous is this gap to the field? (0=trivial, 1=critical)
- information_gain_score: How much would resolving this advance understanding? (0-1)
- confidence_score: How confident are you in this boundary identification? (0-1)"""


class Agent5:
    def __init__(self, api_key: str = None):
        self.client = Mistral(api_key=api_key or os.environ.get("MISTRAL_API_KEY"))
        self.model = "mistral-large-latest"

    def _build_hypotheses_block(self, hypotheses: List[HypothesisInput]) -> str:
        lines = []
        for i, h in enumerate(hypotheses, 1):
            lines.append(f"[H{i}] Paper {h.paper_id}: \"{h.text}\"")
            if h.assumptions:
                lines.append(f"     Assumptions: {', '.join(h.assumptions)}")
            if h.variables:
                lines.append(f"     Variables: {', '.join(h.variables)}")
        return "\n".join(lines)

    def _build_compat_context(self, compat_results: List[Agent4Output]) -> str:
        if not compat_results:
            return "No compatibility data available."
        lines = []
        for r in compat_results:
            short_h1 = r.hypothesis_1[:80] + "..." if len(r.hypothesis_1) > 80 else r.hypothesis_1
            short_h2 = r.hypothesis_2[:80] + "..." if len(r.hypothesis_2) > 80 else r.hypothesis_2
            lines.append(
                f"- [{r.compatibility_type.value}] divergence={r.world_model_divergence_score:.2f}: "
                f"H1: \"{short_h1}\" vs H2: \"{short_h2}\""
            )
            if r.conflict_basis:
                lines.append(f"  Conflict: {'; '.join(r.conflict_basis[:2])}")
        return "\n".join(lines)

    def _detect_boundary(
        self,
        hypotheses: List[HypothesisInput],
        compat_results: List[Agent4Output],
        boundary_index: int
    ) -> Agent5Output:
        hyp_block = self._build_hypotheses_block(hypotheses)
        compat_ctx = self._build_compat_context(compat_results)

        # Focus on high-divergence pairs for targeted boundary analysis
        high_risk = [r for r in compat_results
                     if r.compatibility_type == CompatibilityType.INCOMPATIBLE
                     or r.world_model_divergence_score > 0.6]

        focused_ctx = self._build_compat_context(high_risk) if high_risk else compat_ctx

        prompt = BOUNDARY_PROMPT.format(
            hypotheses_block=hyp_block,
            compat_context=focused_ctx
        )

        response = self.client.chat.complete(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
        )

        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        data = json.loads(raw)

        all_paper_ids = list({h.paper_id for h in hypotheses})

        return Agent5Output(
            boundary_id=f"boundary_{boundary_index}_{str(uuid.uuid4())[:8]}",
            related_hypotheses=[h.text for h in hypotheses],
            boundary_type=BoundaryType(data["boundary_type"]),
            failure_conditions=data.get("failure_conditions", []),
            stress_test_summary=data.get("stress_test_summary", ""),
            unknown_unknown_indicator=bool(data.get("unknown_unknown_indicator", False)),
            research_gap=ResearchGap(
                description=data["research_gap"]["description"],
                reason_unresolved=data["research_gap"]["reason_unresolved"],
                suggested_investigation=data["research_gap"]["suggested_investigation"]
            ),
            epistemic_risk_score=float(data.get("epistemic_risk_score", 0.5)),
            information_gain_score=float(data.get("information_gain_score", 0.5)),
            counterfactual_probes=data.get("counterfactual_probes", []),
            confidence_score=float(data.get("confidence_score", 0.5)),
            source_references=all_paper_ids,
            trace=TRACE
        )

    def analyze(
        self,
        hypotheses: List[HypothesisInput],
        compat_results: List[Agent4Output]
    ) -> List[Agent5Output]:
        """
        Analyze epistemic boundaries given hypotheses and Agent 4 compatibility results.
        Generates one boundary analysis per logical cluster of hypotheses.
        """
        results = []

        # Primary analysis: full cluster
        try:
            primary = self._detect_boundary(hypotheses, compat_results, 0)
            results.append(primary)
        except Exception as e:
            print(f"[Agent5] Primary boundary detection failed: {e}")

        # Secondary: focus on incompatible pairs if they exist
        incompatible_pairs = [
            r for r in compat_results
            if r.compatibility_type == CompatibilityType.INCOMPATIBLE
        ]

        if incompatible_pairs and len(hypotheses) >= 2:
            # Extract unique hypotheses involved in conflicts
            conflict_hyp_texts = set()
            for r in incompatible_pairs:
                conflict_hyp_texts.add(r.hypothesis_1)
                conflict_hyp_texts.add(r.hypothesis_2)

            conflict_hyps = [
                h for h in hypotheses
                if h.text in conflict_hyp_texts
            ]

            if conflict_hyps and len(conflict_hyps) >= 2:
                try:
                    secondary = self._detect_boundary(
                        conflict_hyps,
                        incompatible_pairs,
                        1
                    )
                    results.append(secondary)
                except Exception as e:
                    print(f"[Agent5] Secondary boundary detection failed: {e}")

        return results
