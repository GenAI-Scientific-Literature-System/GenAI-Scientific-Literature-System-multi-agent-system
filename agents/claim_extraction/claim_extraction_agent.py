"""
agents/claim_extraction/claim_extraction_agent.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Agent 1: Claim Extraction

Derived from GenAI Agent 1 (src/agents/agent1_claim.py)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
IDENTITY
  agent_id        : "agent_1_claim_extraction"
  role            : Extract structured scientific claims C=(S,P,O,M,D,Θ)
  genai_origin    : src/agents/agent1_claim.py

GENAI FORMAL MODEL (preserved exactly)
  C = (Subject, Predicate, Object, Method, Domain, Theta)
  Query for RAG: "main claims findings results contributions conclusions"
  Token reduction vs full text: ~80% when RAG retriever is provided.

ANTI-HALLUCINATION
  [V1] filter_hallucinated_claims() — token overlap grounding after extraction.
  Claims not grounded in source text are dropped and counted in metadata.

PIPELINE CONTRACT
  Input  (AgentContext.papers): List[Dict] with "paper_id" + "text"
  Output (AgentResult.payload):
    {
      "claims":    List[Claim as dict],     # grounded, V1-verified claims
      "dropped":   int,                     # V1 hallucination drops
      "per_paper": {paper_id: n_claims}     # per-paper breakdown
    }
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from agents.base.base_agent import AgentContext, AgentResult, BaseAgent
from core.registry.agent_registry import registry

logger = logging.getLogger("agent.agent_1_claim_extraction")

# ── Prompt: derived from GenAI Agent 1's CLAIM_PROMPT (config.py) ─────────────
# Original used: "You extract scientific claims. Return only a JSON object
# with a 'claims' array. No prose."  We preserve + expand that system message
# as the agent's owned template (Unified architecture pattern).
CLAIM_PROMPT = """\
You are Agent 1 — a scientific claim extraction specialist.
Derived from GenAI Agent 1 (src/agents/agent1_claim.py).

Formal claim model: C = (Subject, Predicate, Object, Method, Domain, Theta)

Extract ALL primary scientific claims from the research paper text below.
A claim is a specific, falsifiable assertion: the core finding, hypothesis, or
contribution the paper defends. Do NOT include background motivation.

For each claim return structured JSON with fields:
  subject    : the main entity/population/system being studied
  predicate  : the key relationship or action (active voice, normalised verb)
  object     : the outcome, finding, or target entity
  method     : experimental method, dataset, or technique used
  domain     : scientific domain (NLP, CV, BioMed, RL, Chemistry, Graph, etc.)
  confidence : "high" | "medium" | "low"

Rules:
  - Every span in the claim MUST be traceable to the paper text (V1 guard checks this)
  - Do NOT fabricate entities or results not present in the text
  - Return ONLY valid JSON — no prose outside the object

Paper text:
{text}

Output format:
{{
  "claims": [
    {{
      "subject":    "...",
      "predicate":  "...",
      "object":     "...",
      "method":     "...",
      "domain":     "...",
      "confidence": "high | medium | low"
    }}
  ]
}}
"""

# RAG query used by GenAI Agent 1 (preserved verbatim)
_CLAIM_QUERY = "main claims findings results contributions conclusions"


@registry.register
class ClaimExtractionAgent(BaseAgent):
    """
    Agent 1: Extracts structured scientific claims from paper texts.

    Derived from GenAI Agent 1 (src/agents/agent1_claim.py)
    Wraps: genai_system.src.agents.agent1_claim.extract_claims()

    The original GenAI function is imported and called unchanged (OCP compliance).
    This class adds:
      - BaseAgent interface (run/context/result contract)
      - Multi-paper iteration (pipeline handles multiple papers at once)
      - Graceful per-paper error isolation
    """

    agent_id        = "agent_1_claim_extraction"
    role            = (
        "Extracts structured scientific claims C=(S,P,O,M,D,Θ) from each paper "
        "using Mistral LLM. Uses RAG retriever for ~80% token reduction. "
        "Applies V1 Anti-Hallucination: filter_hallucinated_claims() grounding check."
    )
    prompt_template = CLAIM_PROMPT
    genai_origin    = "src/agents/agent1_claim.py"   # GenAI credit

    required_context_keys: List[str] = []  # First stage — no upstream dependency

    def __init__(self, retriever=None) -> None:
        """
        Parameters
        ----------
        retriever : DocumentRetriever | None
            Optional RAG retriever (from GenAI utils/document_retriever.py).
            When provided, only the most relevant chunks (~900 chars) are sent
            to the LLM instead of the full text, reducing token usage by ~80%.
            (This is the exact same retriever interface as GenAI Agent 1.)
        """
        super().__init__()
        self._retriever = retriever

    def _execute(self, context: AgentContext) -> AgentResult:
        """
        Iterate over all papers in context and extract claims from each.

        Delegates to genai_system.src.agents.agent1_claim.extract_claims()
        which contains the original GenAI logic including:
          - RAG chunk retrieval (if retriever provided)
          - Mistral LLM call with CLAIM_PROMPT
          - V1 hallucination filter: filter_hallucinated_claims()
        """
        # Lazy import: decouples agent framework from LLM stack for testing
        try:
            from genai_system.src.agents.agent1_claim import extract_claims
        except ImportError as exc:
            return self._fail(
                f"[Derived from GenAI Agent 1] Could not import extract_claims: {exc}. "
                "Ensure genai_system is on the Python path."
            )

        papers: List[Dict[str, Any]] = context.papers
        if not papers:
            logger.warning("[agent_1] No papers in context — returning empty claims.")
            return self._ok({"claims": [], "dropped": 0, "per_paper": {}})

        all_claims:   List[Dict]    = []
        total_dropped: int          = 0
        per_paper:    Dict[str, int] = {}

        for paper in papers:
            paper_id = paper.get("paper_id", paper.get("id", "unknown"))
            text     = paper.get("text", paper.get("abstract", ""))

            if not text:
                logger.warning("[agent_1] Paper '%s' has no text — skipping.", paper_id)
                per_paper[paper_id] = 0
                continue

            try:
                # ── Delegate to original GenAI function ──────────────────────
                # extract_claims() contains:
                #   1. RAG retrieval or full-text sanitisation
                #   2. Mistral call with CLAIM_PROMPT
                #   3. V1 hallucination filter: filter_hallucinated_claims()
                claims, dropped = extract_claims(
                    text=text,
                    paper_id=paper_id,
                    retriever=self._retriever,
                )
                serialised = [
                    c.to_dict() if hasattr(c, "to_dict") else vars(c)
                    for c in claims
                ]
                all_claims.extend(serialised)
                total_dropped      += dropped
                per_paper[paper_id] = len(serialised)

                logger.info(
                    "[agent_1] Paper '%s': %d claims extracted, %d dropped (V1 guard).",
                    paper_id, len(serialised), dropped,
                )
            except Exception as exc:
                logger.error("[agent_1] Error processing paper '%s': %s", paper_id, exc)
                per_paper[paper_id] = 0

        return self._ok(
            payload={
                "claims":    all_claims,
                "dropped":   total_dropped,
                "per_paper": per_paper,
            },
            total_papers=len(papers),
            total_claims=len(all_claims),
            hallucination_drops=total_dropped,
            rag_active=self._retriever is not None,
        )
