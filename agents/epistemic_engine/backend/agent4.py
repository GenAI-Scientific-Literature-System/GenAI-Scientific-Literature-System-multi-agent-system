"""
Agent 4: Hypothesis Compatibility & Scientific Discourse Engine
Detects agreement/disagreement via hypothesis simulation and world-model comparison.

OCP FIX:
  [1] Groq fallback — if Mistral SDK call fails (bad key, quota, etc.)
      the same prompt is sent to Groq via its OpenAI-compatible endpoint.
  [2] counterfactual_analysis / conflict_basis coercion — the LLM sometimes
      returns a list of dicts ({scenario:..., implication:...}) instead of
      plain strings. _coerce_str_list() flattens them so pydantic validation
      no longer raises "Input should be a valid string".
"""

import json
import os
import itertools
import requests
import logging
from typing import List, Dict, Any

from models import (
    Agent4Output, CompatibilityType, TraceInfo, HypothesisInput
)

logger = logging.getLogger(__name__)

TRACE = TraceInfo(component="Agent_4", method="hypothesis_simulation")

SYSTEM_PROMPT = """You are a AAAI-level scientific reasoning engine specializing in 
hypothesis compatibility analysis. Your task is to simulate whether two scientific 
hypotheses can coexist in the same logical/empirical world.

Approach:
1. Canonicalize each hypothesis into structured form (variables, assumptions, outcomes)
2. Simulate a "possible world" where both hypotheses are true simultaneously
3. Detect logical, empirical, or contextual contradictions
4. Classify the compatibility relationship
5. Identify the root cause of any conflict

Be precise, rigorous, and justify every claim with reasoning.
You MUST respond with valid JSON only — no preamble, no markdown fences."""

SIMULATION_PROMPT = """Analyze the compatibility of these two scientific hypotheses:

HYPOTHESIS 1 (from paper {paper_id_1}):
"{hypothesis_1}"
Assumptions: {assumptions_1}
Variables: {variables_1}
Evidence: {evidence_1}

HYPOTHESIS 2 (from paper {paper_id_2}):
"{hypothesis_2}"
Assumptions: {assumptions_2}
Variables: {variables_2}
Evidence: {evidence_2}

Additional context: {context}

Perform a hypothesis simulation:
1. Can both hypotheses be true in the same world simultaneously?
2. What are the conditions (if any) under which they could coexist?
3. What causes incompatibility (if any)?
4. Generate counterfactual scenarios.

Respond with this exact JSON structure:
{{
  "compatibility_type": "COEXISTENT|CONDITIONALLY_COMPATIBLE|INCOMPATIBLE|UNKNOWN",
  "simulation_summary": "2-3 sentence description of the simulation result",
  "conflict_basis": ["string reason 1", "string reason 2"],
  "world_model_divergence_score": 0.0,
  "counterfactual_analysis": ["string scenario 1", "string scenario 2", "string scenario 3"],
  "confidence_score": 0.0
}}

IMPORTANT: conflict_basis and counterfactual_analysis must be plain string lists, NOT objects.

Scoring rules:
- world_model_divergence_score: 0.0 = same world, 1.0 = completely incompatible worlds
- confidence_score: based on evidence quality and logical clarity (0-1)
- Be specific in conflict_basis — name the exact variable, assumption, or measurement at odds"""


def _coerce_str_list(raw) -> List[str]:
    """
    Ensure every element of a list is a plain string.
    The LLM sometimes returns [{"scenario": "...", "implication": "..."}, ...]
    instead of ["...", "..."] — flatten those dicts to a readable string.
    """
    if not isinstance(raw, list):
        return []
    result = []
    for item in raw:
        if isinstance(item, str):
            result.append(item)
        elif isinstance(item, dict):
            # Join all values in the dict into one string
            parts = [str(v) for v in item.values() if v]
            result.append(" — ".join(parts) if parts else str(item))
        else:
            result.append(str(item))
    return result


def _parse_response(raw: str, h1: HypothesisInput, h2: HypothesisInput) -> Agent4Output:
    """Parse LLM JSON response into Agent4Output, coercing list fields."""
    # Strip markdown fences
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    data = json.loads(raw)

    return Agent4Output(
        hypothesis_1=h1.text,
        hypothesis_2=h2.text,
        compatibility_type=CompatibilityType(data.get("compatibility_type", "UNKNOWN")),
        simulation_summary=data.get("simulation_summary", ""),
        conflict_basis=_coerce_str_list(data.get("conflict_basis") or []),
        world_model_divergence_score=float(data.get("world_model_divergence_score", 0.5)),
        counterfactual_analysis=_coerce_str_list(data.get("counterfactual_analysis") or []),
        confidence_score=float(data.get("confidence_score", 0.5)),
        source_references=[h1.paper_id, h2.paper_id],
        trace=TRACE,
    )


class Agent4:
    def __init__(self, api_key: str = None):
        self.mistral_key = api_key or os.environ.get("MISTRAL_API_KEY", "")
        self.groq_key = (
            os.environ.get("GROQ_API_KEY") or
            os.environ.get("GROQ_API_KEY_1") or ""
        )
        self.mistral_model = "mistral-large-latest"
        self.groq_model = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

    def _call_mistral_sdk(self, prompt: str) -> str:
        from mistralai import Mistral
        client = Mistral(api_key=self.mistral_key)
        response = client.chat.complete(
            model=self.mistral_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=1000,
        )
        return response.choices[0].message.content.strip()

    def _call_groq_http(self, prompt: str) -> str:
        if not self.groq_key:
            raise RuntimeError("No GROQ_API_KEY available for fallback.")
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.groq_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.groq_model,
                "max_tokens": 1000,
                "temperature": 0.1,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
            },
            timeout=45,
        )
        # Some Groq models don't support json_object — retry without it
        if resp.status_code == 400:
            resp = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.groq_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.groq_model,
                    "max_tokens": 1000,
                    "temperature": 0.1,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": prompt},
                    ],
                },
                timeout=45,
            )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()

    def _canonicalize(self, hyp: HypothesisInput) -> Dict[str, Any]:
        return {
            "id":          hyp.id,
            "text":        hyp.text,
            "paper_id":    hyp.paper_id,
            "assumptions": hyp.assumptions or [],
            "variables":   hyp.variables or [],
            "evidence":    hyp.evidence or "Not provided",
        }

    def _simulate_pair(
        self,
        h1: HypothesisInput,
        h2: HypothesisInput,
        context: str = "",
    ) -> Agent4Output:
        c1 = self._canonicalize(h1)
        c2 = self._canonicalize(h2)

        prompt = SIMULATION_PROMPT.format(
            paper_id_1   = c1["paper_id"],
            hypothesis_1 = c1["text"],
            assumptions_1= ", ".join(c1["assumptions"]) or "Not specified",
            variables_1  = ", ".join(c1["variables"])   or "Not specified",
            evidence_1   = c1["evidence"],
            paper_id_2   = c2["paper_id"],
            hypothesis_2 = c2["text"],
            assumptions_2= ", ".join(c2["assumptions"]) or "Not specified",
            variables_2  = ", ".join(c2["variables"])   or "Not specified",
            evidence_2   = c2["evidence"],
            context      = context or "General scientific domain",
        )

        raw = None
        last_error = None

        # ── Try Mistral first ─────────────────────────────────────────────────
        if self.mistral_key:
            try:
                raw = self._call_mistral_sdk(prompt)
            except Exception as e:
                last_error = e
                logger.warning("Agent4 Mistral failed (%s) — trying Groq.", e)

        # ── Groq fallback ─────────────────────────────────────────────────────
        if raw is None:
            try:
                raw = self._call_groq_http(prompt)
            except Exception as e:
                last_error = e
                logger.warning("Agent4 Groq also failed: %s", e)

        if raw is None:
            raise RuntimeError(f"All LLM providers failed. Last error: {last_error}")

        return _parse_response(raw, h1, h2)

    def simulate(
        self,
        hypotheses: List[HypothesisInput],
        context: str = "",
    ) -> List[Agent4Output]:
        results = []
        for h1, h2 in itertools.combinations(hypotheses, 2):
            try:
                results.append(self._simulate_pair(h1, h2, context))
            except Exception as e:
                results.append(Agent4Output(
                    hypothesis_1=h1.text,
                    hypothesis_2=h2.text,
                    compatibility_type=CompatibilityType.UNKNOWN,
                    simulation_summary=f"Simulation failed: {str(e)}",
                    conflict_basis=[],
                    world_model_divergence_score=0.5,
                    counterfactual_analysis=[],
                    confidence_score=0.0,
                    source_references=[h1.paper_id, h2.paper_id],
                    trace=TRACE,
                ))
        return results
