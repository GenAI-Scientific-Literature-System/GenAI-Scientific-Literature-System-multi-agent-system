"""
MERLIN Data Schemas
Formal definitions from the paper: C=(S,P,O,M,D,Θ), A=(type,scope,...), R(Ci,Cj|Ai,Aj)
"""
from dataclasses import dataclass, field, asdict
from typing import Optional, List
from enum import Enum
import uuid


class RelationType(str, Enum):
    AGREE        = "agree"
    CONTRADICT   = "contradict"
    CONDITIONAL  = "conditional"
    UNRELATED    = "unrelated"


class AssumptionType(str, Enum):
    DOMAIN      = "domain"
    METHOD      = "method"
    SCOPE       = "scope"
    STATISTICAL = "statistical"


class VerificationStatus(str, Enum):
    VERIFIED = "VERIFIED"
    WEAK     = "WEAK"
    REJECTED = "REJECTED"


class GapType(str, Enum):
    THEORETICAL   = "theoretical"
    EMPIRICAL     = "empirical"
    METHODOLOGICAL = "methodological"


# ── Assumption  A=(type, scope, constraint, explicitness, evidence_span) ──────
@dataclass
class Assumption:
    id: str                = field(default_factory=lambda: str(uuid.uuid4())[:8])
    type: str              = AssumptionType.DOMAIN
    constraint: str        = ""
    explicit: bool         = True
    span: str              = ""
    verification: str      = VerificationStatus.WEAK
    score: float           = 0.0

    def to_dict(self):
        return asdict(self)


# ── Claim  C=(S, P, O, M, D, Θ) ──────────────────────────────────────────────
@dataclass
class Claim:
    id: str                   = field(default_factory=lambda: str(uuid.uuid4())[:8])
    subject: str              = ""
    predicate: str            = ""
    object: str               = ""
    method: str               = ""
    domain: str               = ""
    theta: dict               = field(default_factory=dict)        # Θ: hyperparams
    evidence_spans: List[str] = field(default_factory=list)
    evidence_strength: str    = "medium"
    assumptions: List[Assumption] = field(default_factory=list)
    uncertainty: float        = 0.0
    paper_id: str             = ""

    @property
    def text(self):
        return f"{self.subject} {self.predicate} {self.object}"

    def to_dict(self):
        d = asdict(self)
        d['text'] = self.text
        return d


# ── Agreement  R(Ci, Cj | Ai, Aj) ────────────────────────────────────────────
@dataclass
class Agreement:
    claim_i_id: str
    claim_j_id: str
    relation: str       = RelationType.UNRELATED
    confidence: float   = 0.0
    reason: str         = ""
    assumption_overlap: float = 0.0
    agreement_basis: str  = ""  # "identical-sets|disjoint|partial|predicate|path"
    shared_assumptions: list = field(default_factory=list)  # IDs in A1∩A2

    def to_dict(self):
        return asdict(self)


# ── Research Gap ──────────────────────────────────────────────────────────────
@dataclass
class ResearchGap:
    id: str           = field(default_factory=lambda: str(uuid.uuid4())[:8])
    gap: str          = ""
    type: str         = GapType.EMPIRICAL
    priority: str     = "medium"
    related_claims: List[str] = field(default_factory=list)
    uncertainty_score: float  = 0.0
    gap_signals: dict = field(default_factory=dict)  # {degree, bc, uncertainty, evidence}

    def to_dict(self):
        return asdict(self)


# ── EDG Node ──────────────────────────────────────────────────────────────────
@dataclass
class EDGNode:
    node_id: str
    node_type: str    # "claim" | "assumption"
    data: dict        = field(default_factory=dict)
    uncertainty: float = 0.0
