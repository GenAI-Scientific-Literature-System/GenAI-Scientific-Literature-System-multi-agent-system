"""
config.py  —  OCP Extension: configuration module for genai_system/src agents.

Uses LazyEnvStr so every access re-reads os.environ at use-time, not import-time.
This means API keys set after import (e.g. loaded from .env by run.bat) are
picked up correctly - fixing the 401 Unauthorized error.
"""
import os


class _LazyEnvStr(str):
    """str subclass that re-reads os.environ on every string operation."""
    def __new__(cls, env_key, default=""):
        instance = super().__new__(cls, os.environ.get(env_key, default))
        instance._env_key = env_key
        instance._default = default
        return instance
    def _f(self): return os.environ.get(self._env_key, self._default)
    def __str__(self):         return self._f()
    def __repr__(self):        return repr(self._f())
    def __len__(self):         return len(self._f())
    def __bool__(self):        return bool(self._f())
    def __eq__(self, o):       return self._f() == o
    def __hash__(self):        return hash(self._f())
    def __add__(self, o):      return self._f() + o
    def __radd__(self, o):     return o + self._f()
    def __contains__(self, i): return i in self._f()
    def __format__(self, s):   return format(self._f(), s)
    def format(self, *a, **k): return self._f().format(*a, **k)
    def strip(self, *a):       return self._f().strip(*a)
    def lower(self):           return self._f().lower()
    def upper(self):           return self._f().upper()
    def split(self, *a, **k):  return self._f().split(*a, **k)
    def startswith(self, *a):  return self._f().startswith(*a)
    def endswith(self, *a):    return self._f().endswith(*a)
    def replace(self, *a):     return self._f().replace(*a)
    def encode(self, *a, **k): return self._f().encode(*a, **k)


class _LazyEnvFloat(float):
    """float subclass that re-reads os.environ on every numeric operation."""
    def __new__(cls, env_key, default):
        val = float(os.environ.get(env_key, str(default)))
        i = super().__new__(cls, val)
        i._env_key = env_key; i._default = default
        return i
    def _f(self):
        try: return float(os.environ.get(self._env_key, str(self._default)))
        except: return self._default
    def __float__(self): return self._f()
    def __repr__(self):  return repr(self._f())
    def __str__(self):   return str(self._f())
    def __eq__(self, o): return self._f() == o
    def __lt__(self, o): return self._f() < o
    def __le__(self, o): return self._f() <= o
    def __gt__(self, o): return self._f() > o
    def __ge__(self, o): return self._f() >= o


class _LazyEnvInt(int):
    """int subclass that re-reads os.environ on every numeric operation."""
    def __new__(cls, env_key, default):
        val = int(os.environ.get(env_key, str(default)))
        i = super().__new__(cls, val)
        i._env_key = env_key; i._default = default
        return i
    def _f(self):
        try: return int(os.environ.get(self._env_key, str(self._default)))
        except: return self._default
    def __int__(self):   return self._f()
    def __index__(self): return self._f()
    def __repr__(self):  return repr(self._f())
    def __str__(self):   return str(self._f())
    def __eq__(self, o): return self._f() == o
    def __lt__(self, o): return self._f() < o
    def __le__(self, o): return self._f() <= o
    def __gt__(self, o): return self._f() > o
    def __ge__(self, o): return self._f() >= o


# Mistral AI
MISTRAL_API_KEY    = _LazyEnvStr("MISTRAL_API_KEY", "")
MISTRAL_MODEL      = _LazyEnvStr("MISTRAL_MODEL", "mistral-small-latest")
MISTRAL_MAX_TOKENS = _LazyEnvInt("MISTRAL_MAX_TOKENS", 1024)
MISTRAL_TEMP       = _LazyEnvFloat("MISTRAL_TEMP", 0.1)

# Groq (fallback when Mistral is unavailable)
GROQ_API_KEY   = _LazyEnvStr("GROQ_API_KEY", "")
GROQ_MODEL     = _LazyEnvStr("GROQ_MODEL", "llama-3.3-70b-versatile")

# NLI (Agent 6.1)
NLI_THRESHOLD      = _LazyEnvFloat("NLI_THRESHOLD", 0.5)
NLI_MODEL_PRIMARY  = _LazyEnvStr("NLI_MODEL_PRIMARY", "cross-encoder/nli-deberta-v3-small")

# Agent 5 uncertainty weights
UNCERTAINTY_CONFLICT_WEIGHT  = _LazyEnvFloat("UNCERTAINTY_CONFLICT_WEIGHT",  0.30)
UNCERTAINTY_EVIDENCE_WEIGHT  = _LazyEnvFloat("UNCERTAINTY_EVIDENCE_WEIGHT",  0.20)
UNCERTAINTY_STABILITY_WEIGHT = _LazyEnvFloat("UNCERTAINTY_STABILITY_WEIGHT", 0.30)

# Agent prompts
CLAIM_PROMPT = """You are a scientific claim extraction agent. Given the text of a research paper,
extract the single most central scientific claim.

Return ONLY a JSON object with a 'claims' array. Each item must have:
  subject, predicate, object, method, domain, uncertainty (0.0-1.0), assumptions (list of strings)

Paper text:
{text}

Example format:
{{
  "claims": [
    {{
      "subject": "LLMs",
      "predicate": "outperform",
      "object": "BM25 on QA tasks",
      "method": "fine-tuning",
      "domain": "information retrieval",
      "uncertainty": 0.2,
      "assumptions": ["sufficient training data", "standard benchmarks"]
    }}
  ]
}}"""

EVIDENCE_PROMPT = """You are a scientific evidence attribution agent.
Given a list of claims and a paper text, identify supporting evidence spans for each claim.

Return ONLY a JSON object with an 'evidence' array. Each item must have:
  claim_index (int), evidence_span (string), confidence (0.0-1.0), relation (SUPPORTS|CONTRADICTS|INCONCLUSIVE)

Claims:
{claims}

Paper text:
{text}

Example format:
{{
  "evidence": [
    {{
      "claim_index": 0,
      "evidence_span": "Results show 12% F1 improvement over BM25 baseline",
      "confidence": 0.9,
      "relation": "SUPPORTS"
    }}
  ]
}}"""

ASSUMPTION_PROMPT = """You are a scientific assumption extraction agent.
Given a paper text, identify all assumptions, constraints, and limitations.

Return ONLY a JSON object with an 'assumptions' array. Each item must have:
  text (string), type (METHODOLOGICAL|STATISTICAL|DOMAIN|IMPLICIT), scope (string), explicitness (EXPLICIT|IMPLICIT)

Paper text:
{text}

Example format:
{{
  "assumptions": [
    {{
      "text": "Standard benchmark datasets are representative",
      "type": "DOMAIN",
      "scope": "evaluation",
      "explicitness": "IMPLICIT"
    }}
  ]
}}"""
