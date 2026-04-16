"""
Pipeline Coordinator: Agent 4 → Agent 5 → Aggregation
"""

from typing import Dict, Any, List
from models import PipelineInput, PipelineOutput, Agent4Output, CompatibilityType
from agent4 import Agent4
from agent5 import Agent5


def compute_summary(compat_results: List[Agent4Output]) -> Dict[str, Any]:
    """Aggregate pipeline-level statistics."""
    if not compat_results:
        return {}

    type_counts = {t.value: 0 for t in CompatibilityType}
    for r in compat_results:
        type_counts[r.compatibility_type.value] += 1

    total = len(compat_results)
    avg_divergence = sum(r.world_model_divergence_score for r in compat_results) / total
    avg_confidence = sum(r.confidence_score for r in compat_results) / total

    # Consensus strength: low divergence = strong consensus
    consensus_strength = 1.0 - avg_divergence

    return {
        "total_pairs_analyzed": total,
        "compatibility_distribution": type_counts,
        "average_divergence_score": round(avg_divergence, 3),
        "average_confidence": round(avg_confidence, 3),
        "consensus_strength": round(consensus_strength, 3),
        "high_conflict_count": type_counts.get("INCOMPATIBLE", 0),
        "stable_consensus_count": type_counts.get("COEXISTENT", 0)
    }


def run_pipeline(input_data: PipelineInput) -> PipelineOutput:
    """
    Execute the full Agent 4 → Agent 5 pipeline.
    
    Flow:
    1. Agent 4 simulates pairwise hypothesis compatibility
    2. Agent 5 consumes compatibility results + original hypotheses
       to detect epistemic boundaries and research gaps
    3. Aggregate summary statistics
    """
    agent4 = Agent4()
    agent5 = Agent5()

    # Stage 1: Hypothesis Compatibility (Agent 4)
    print(f"[Pipeline] Running Agent 4 on {len(input_data.hypotheses)} hypotheses...")
    compat_results = agent4.simulate(
        input_data.hypotheses,
        context=input_data.context or ""
    )
    print(f"[Pipeline] Agent 4 complete: {len(compat_results)} pairs analyzed")

    # Stage 2: Epistemic Boundary Analysis (Agent 5)
    print("[Pipeline] Running Agent 5...")
    uncertainty_results = agent5.analyze(
        input_data.hypotheses,
        compat_results
    )
    print(f"[Pipeline] Agent 5 complete: {len(uncertainty_results)} boundaries detected")

    # Stage 3: Aggregate
    summary = compute_summary(compat_results)

    return PipelineOutput(
        agreements=compat_results,
        uncertainties=uncertainty_results,
        summary=summary
    )
