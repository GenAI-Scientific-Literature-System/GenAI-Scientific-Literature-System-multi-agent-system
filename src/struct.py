"""
MERLINStruct — The Primary Reasoning Representation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
After extraction, raw text is discarded. ALL downstream reasoning
(Agent 4, Agent 5, EDG, gap detection) operates ONLY on this struct.

Format:
{
  "claims": {
    "C1": {"pred": "outperforms", "subj": "LLM", "obj": "BM25", "domain": "NLP",
           "assumptions": ["A1", "A2"], "uncertainty": 0.0},
    ...
  },
  "assumptions": {
    "A1": {"type": "method", "constraint": "high-resource GPU", "explicit": true},
    ...
  },
  "relations": [
    {"i": "C1", "j": "C2", "relation": "contradict", "confidence": 0.8}
  ]
}

Token cost of full text  → ~1800 chars  → ~450 tokens per paper
Token cost of this struct → ~200 chars  → ~50 tokens for reasoning
Reduction: 90 %+
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class MERLINStruct:
    """
    The sole input to all reasoning agents after the extraction phase.
    Text is never passed beyond build().
    """
    claims:      Dict[str, dict] = field(default_factory=dict)
    # { claim_id: {pred, subj, obj, domain, assumptions:[aid,...], uncertainty} }

    assumptions: Dict[str, dict] = field(default_factory=dict)
    # { assumption_id: {type, constraint, explicit} }

    relations:   List[dict]      = field(default_factory=list)
    # [{i, j, relation, confidence, reason}]  — filled by Agent 4

    # ── Builders ──────────────────────────────────────────────────────────────

    @classmethod
    def build(cls, claims, assumptions) -> "MERLINStruct":
        """
        Convert Claim / Assumption dataclass lists → compact struct.
        Called at end of extraction phase. After this, claim objects
        are only used for uncertainty propagation; text is never re-sent.
        """
        s = cls()

        for a in assumptions:
            s.assumptions[a.id] = {
                "type":       a.type,
                "constraint": a.constraint,
                "explicit":   a.explicit,
            }

        for c in claims:
            s.claims[c.id] = {
                "pred":        c.predicate,
                "subj":        c.subject,
                "obj":         c.object,
                "domain":      c.domain,
                "assumptions": [a.id for a in c.assumptions],
                "uncertainty": c.uncertainty,
                "paper_id":    c.paper_id,
            }

        logger.info(
            "MERLINStruct built: %d claims, %d assumptions — text discarded.",
            len(s.claims), len(s.assumptions),
        )
        return s

    # ── Accessors (used by reasoning agents) ─────────────────────────────────

    def assumption_set(self, claim_id: str) -> set:
        """
        Returns the SET of assumption constraint tokens for a claim.
        Used for set-operation assumption reasoning.
        """
        import re
        aids = self.claims.get(claim_id, {}).get("assumptions", [])
        tokens = set()
        for aid in aids:
            constraint = self.assumptions.get(aid, {}).get("constraint", "")
            tokens |= {t.lower() for t in re.split(r"\W+", constraint) if len(t) > 3}
        return tokens

    def assumption_types(self, claim_id: str) -> set:
        """Returns the set of assumption TYPES for a claim."""
        aids = self.claims.get(claim_id, {}).get("assumptions", [])
        return {self.assumptions.get(aid, {}).get("type", "") for aid in aids}

    def id_prompt(self) -> str:
        """
        Minimal ID-based JSON representation for LLM prompts.
        Sends ~50 tokens instead of ~450.

        Format: {"C1": {"p":"outperforms","a":["A1","A2"]}, ...}
                {"A1": "high-resource GPU", ...}
        """
        import json
        c_map = {
            cid: {"p": d["pred"], "s": d["subj"][:20], "o": d["obj"][:20],
                  "a": d["assumptions"]}
            for cid, d in self.claims.items()
        }
        a_map = {
            aid: d["constraint"][:30]
            for aid, d in self.assumptions.items()
        }
        return json.dumps({"C": c_map, "A": a_map}, separators=(",", ":"))

    def add_relation(self, i: str, j: str, relation: str,
                     confidence: float, reason: str = ""):
        self.relations.append({
            "i": i, "j": j,
            "relation":   relation,
            "confidence": round(confidence, 3),
            "reason":     reason,
        })

    def to_dict(self) -> dict:
        return {
            "claims":      self.claims,
            "assumptions": self.assumptions,
            "relations":   self.relations,
        }
