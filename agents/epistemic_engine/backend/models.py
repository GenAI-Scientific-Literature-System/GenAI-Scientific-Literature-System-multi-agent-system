from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum


class CompatibilityType(str, Enum):
    COEXISTENT = "COEXISTENT"
    CONDITIONALLY_COMPATIBLE = "CONDITIONALLY_COMPATIBLE"
    INCOMPATIBLE = "INCOMPATIBLE"
    UNKNOWN = "UNKNOWN"


class BoundaryType(str, Enum):
    GENERALIZATION_FAILURE = "GENERALIZATION_FAILURE"
    ASSUMPTION_BREAKDOWN = "ASSUMPTION_BREAKDOWN"
    UNEXPLORED_DIMENSION = "UNEXPLORED_DIMENSION"


class TraceInfo(BaseModel):
    component: str
    method: str


class Agent4Output(BaseModel):
    hypothesis_1: str
    hypothesis_2: str
    compatibility_type: CompatibilityType
    simulation_summary: str
    conflict_basis: List[str]
    world_model_divergence_score: float = Field(ge=0.0, le=1.0)
    counterfactual_analysis: List[str]
    confidence_score: float = Field(ge=0.0, le=1.0)
    source_references: List[str]
    trace: TraceInfo


class ResearchGap(BaseModel):
    description: str
    reason_unresolved: str
    suggested_investigation: str


class Agent5Output(BaseModel):
    boundary_id: str
    related_hypotheses: List[str]
    boundary_type: BoundaryType
    failure_conditions: List[str]
    stress_test_summary: str
    unknown_unknown_indicator: bool
    research_gap: ResearchGap
    epistemic_risk_score: float = Field(ge=0.0, le=1.0)
    information_gain_score: float = Field(ge=0.0, le=1.0)
    counterfactual_probes: List[str]
    confidence_score: float = Field(ge=0.0, le=1.0)
    source_references: List[str]
    trace: TraceInfo


class HypothesisInput(BaseModel):
    id: str
    text: str
    paper_id: str
    domain: Optional[str] = None
    assumptions: Optional[List[str]] = []
    variables: Optional[List[str]] = []
    evidence: Optional[str] = None


class PipelineInput(BaseModel):
    hypotheses: List[HypothesisInput]
    context: Optional[str] = None


class PipelineOutput(BaseModel):
    agreements: List[Agent4Output]
    uncertainties: List[Agent5Output]
    summary: Dict[str, Any]
