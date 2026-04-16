"""
Agent 3: Normalisation
Standardises claim predicates and domains for cross-paper comparison.
Uses rule-based mapping first (zero tokens), falls back to Mistral only for ambiguous cases.
"""
import re
import logging
from typing import List
from src.models.schemas import Claim

logger = logging.getLogger(__name__)

# Rule-based predicate normalisation (no LLM tokens needed)
PREDICATE_MAP = {
    r"outperform|surpass|exceed|beat":           "outperforms",
    r"improve|enhance|boost|increase":           "improves",
    r"reduce|decrease|lower|minimize":           "reduces",
    r"cause|lead to|result in|produce":          "causes",
    r"correlate|associate|relate":               "correlates_with",
    r"fail|underperform|degrade":                "underperforms",
    r"propose|introduce|present|describe":       "proposes",
    r"show|demonstrate|prove|establish":         "demonstrates",
}

DOMAIN_MAP = {
    r"nlp|natural language|text|language model": "NLP",
    r"vision|image|visual|cv":                   "CV",
    r"reinforcement|rl|reward":                  "RL",
    r"biomedical|clinical|medical|health":       "BioMed",
    r"chemistry|drug|molecule|protein":          "Chemistry",
    r"graph|network|knowledge base":             "Graph",
}


def _normalise_field(text: str, mapping: dict) -> str:
    t = text.lower().strip()
    for pattern, norm in mapping.items():
        if re.search(pattern, t):
            return norm
    return text  # keep original if no match


def normalise_claims(claims: List[Claim]) -> List[Claim]:
    """
    Rule-based normalisation — zero Mistral tokens.
    Normalises predicates and domains for consistency.
    """
    for c in claims:
        c.predicate = _normalise_field(c.predicate, PREDICATE_MAP)
        c.domain    = _normalise_field(c.domain,    DOMAIN_MAP)
    logger.info("Agent 3: Normalised %d claims (rule-based, 0 tokens).", len(claims))
    return claims
