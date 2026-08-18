from typing import List, Dict, Any
from src.agents.agent4_agreement import compute_agreements
from src.models.schemas import Claim

class AgreementDetector:
    def __init__(self, *args, **kwargs):
        pass

    def detect(self, claims: List[Any]) -> List[Dict[str, Any]]:
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
        agreements, _ = compute_agreements(typed_claims)
        return [a.to_dict() if hasattr(a, 'to_dict') else a for a in agreements]
