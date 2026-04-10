"""
agents/base/base_agent.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BaseAgent — Abstract foundation for every agent in the unified pipeline.

DESIGN PRINCIPLES:
  ┌─────────────────────────────────────────────────────────────────────┐
  │  OCP  : Open for extension (new agents), closed for modification    │
  │  LSP  : All concrete agents are substitutable for BaseAgent         │
  │  ISP  : Thin interface — agents only implement what they need        │
  │  DIP  : Orchestrator depends on abstraction, not concrete classes    │
  └─────────────────────────────────────────────────────────────────────┘

AGENT CONTRACT:
  Every agent MUST declare:
    - agent_id        : unique machine-readable identifier
    - role            : human-readable role description
    - prompt_template : the LLM prompt this agent owns (None for rule-based)
    - genai_origin    : original GenAI file this agent derives from

WHAT CHANGED vs original refactored base (extensions only — OCP):
  + `genai_origin`         class attribute  — preserves GenAI credit lineage
  + `get_prompt_context()` helper           — renders prompt_template safely
  + `_assumption_verify()` hook             — invokes GenAI Agent 6.1 V5 guard
  + `to_info_dict()`                        — exposes agent metadata to gateway
"""
from __future__ import annotations

import abc
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Data contracts shared by ALL agents
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class AgentContext:
    """Input contract passed into every agent's run(). Read-only by agents."""
    query:    str                    = ""
    papers:   List[Dict[str, Any]]   = field(default_factory=list)
    pipeline: Dict[str, Any]         = field(default_factory=dict)
    metadata: Dict[str, Any]         = field(default_factory=dict)

    def upstream(self, agent_id: str, key: str, default: Any = None) -> Any:
        """Fetch a specific key from a previous agent's output."""
        return self.pipeline.get(agent_id, {}).get(key, default)


@dataclass
class AgentResult:
    """Typed, traceable output contract returned by every agent's run()."""
    agent_id:     str                  = ""
    role:         str                  = ""
    genai_origin: str                  = ""   # NEW — GenAI provenance credit
    success:      bool                 = True
    payload:      Dict[str, Any]       = field(default_factory=dict)
    error:        Optional[str]        = None
    elapsed_sec:  float                = 0.0
    metadata:     Dict[str, Any]       = field(default_factory=dict)

    @classmethod
    def failure(cls, agent_id, role, error, elapsed_sec=0.0, genai_origin=""):
        return cls(agent_id=agent_id, role=role, genai_origin=genai_origin,
                   success=False, payload={}, error=error, elapsed_sec=elapsed_sec)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id":     self.agent_id,
            "role":         self.role,
            "genai_origin": self.genai_origin,
            "success":      self.success,
            "payload":      self.payload,
            "error":        self.error,
            "elapsed_sec":  round(self.elapsed_sec, 3),
            "metadata":     self.metadata,
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BaseAgent
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class BaseAgent(abc.ABC):
    """Abstract base class for all pipeline agents."""

    agent_id:        str           = ""
    role:            str           = ""
    prompt_template: Optional[str] = None
    genai_origin:    str           = ""   # e.g. "src/agents/agent1_claim.py"

    required_context_keys: List[str] = []

    def __init__(self) -> None:
        if not self.agent_id:
            raise TypeError(f"{type(self).__name__} must define a non-empty `agent_id`.")
        self._logger = logging.getLogger(f"agent.{self.agent_id}")

    # FINAL — do not override in subclasses
    def run(self, context: AgentContext) -> AgentResult:
        self._logger.info("[%s] Starting | genai_origin: %s", self.agent_id, self.genai_origin or "N/A")
        t0 = time.perf_counter()
        missing = self._validate_context(context)
        if missing:
            elapsed = time.perf_counter() - t0
            msg = f"Missing required context keys: {missing}"
            self._logger.error("[%s] %s", self.agent_id, msg)
            return AgentResult.failure(self.agent_id, self.role, msg, elapsed, self.genai_origin)
        try:
            result = self._execute(context)
            elapsed = time.perf_counter() - t0
            result.agent_id     = self.agent_id
            result.role         = self.role
            result.genai_origin = self.genai_origin
            result.elapsed_sec  = elapsed
            self._logger.info("[%s] Done %.3fs success=%s", self.agent_id, elapsed, result.success)
            return result
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            self._logger.exception("[%s] Unhandled exception", self.agent_id)
            return AgentResult.failure(self.agent_id, self.role, f"{type(exc).__name__}: {exc}",
                                       elapsed, self.genai_origin)

    @abc.abstractmethod
    def _execute(self, context: AgentContext) -> AgentResult: ...

    # ── NEW helpers ────────────────────────────────────────────────────────────

    def get_prompt_context(self, **kwargs: Any) -> str:
        """Render prompt_template with kwargs. Returns '' for rule-based agents."""
        if not self.prompt_template:
            return ""
        try:
            return self.prompt_template.format(**kwargs)
        except KeyError as exc:
            self._logger.warning("[%s] prompt key missing: %s", self.agent_id, exc)
            return self.prompt_template

    def _assumption_verify(self, assumptions: list, source_text: str) -> list:
        """
        Opt-in V5 Anti-Hallucination guard (GenAI Agent 6.1).
        Call from _execute() when producing Assumption objects.
        """
        try:
            from genai_system.src.agents.agent6_1_verify import verify_all_assumptions
            return verify_all_assumptions(assumptions, source_text)
        except Exception as exc:
            self._logger.warning("[%s] V5 guard unavailable: %s", self.agent_id, exc)
            return assumptions

    def to_info_dict(self) -> Dict[str, Any]:
        """Serialisable agent metadata for gateway /api/pipeline-info."""
        return {
            "agent_id":        self.agent_id,
            "role":            self.role,
            "genai_origin":    self.genai_origin,
            "has_llm_prompt":  self.prompt_template is not None,
            "required_inputs": self.required_context_keys,
        }

    def _validate_context(self, context: AgentContext) -> List[str]:
        missing = []
        for key in self.required_context_keys:
            parts = key.split(".", 1)
            if len(parts) == 2:
                upstream_id, field_name = parts
                if field_name not in context.pipeline.get(upstream_id, {}):
                    missing.append(key)
        return missing

    def _ok(self, payload: Dict[str, Any], **meta) -> AgentResult:
        return AgentResult(success=True, payload=payload, metadata=meta)

    def _fail(self, error: str, **meta) -> AgentResult:
        return AgentResult(success=False, payload={}, error=error, metadata=meta)

    def __repr__(self) -> str:
        return f"<{type(self).__name__} agent_id={self.agent_id!r} genai_origin={self.genai_origin!r}>"
