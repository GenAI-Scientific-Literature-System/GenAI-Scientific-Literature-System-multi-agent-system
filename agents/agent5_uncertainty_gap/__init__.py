from typing import List, Dict, Any
from src.agents.agent5_uncertainty import propagate_uncertainty, detect_gaps
from src.models.schemas import Claim
from src.graph.edg import EpistemicDependencyGraph

class UncertaintyDetector:
    def __init__(self, *args, **kwargs):
        pass

    def detect(self, claims: List[Any], evidence: Any = None) -> List[Dict[str, Any]]:
        if not claims:
            return []
        typed_claims = [
            Claim(
                id=c.get('id', f'c{i}'),
                subject=c.get('subject', ''),
                predicate=c.get('predicate', ''),
                object=c.get('object', ''),
                domain=c.get('domain', ''),
                paper_id=c.get('paper_id', '')
            ) if isinstance(c, dict) else c
            for i, c in enumerate(claims)
        ]
        edg = EpistemicDependencyGraph()
        for c in typed_claims:
            edg.add_claim(c)
        gaps, _ = detect_gaps(typed_claims, edg)
        return [g.to_dict() if hasattr(g, 'to_dict') else g for g in gaps]
