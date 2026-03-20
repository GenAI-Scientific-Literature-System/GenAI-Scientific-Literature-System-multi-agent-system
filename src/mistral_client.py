"""
MERLIN Mistral AI Client
Fixes:
  [1] sanitize_for_prompt() — strips chars that break JSON output
  [2] _repair_json()        — recovers partial/truncated responses
  [3] max_tokens raised     — prevents mid-string cutoff
  [4] array wrapping        — json_object mode needs an object not bare array
"""
import json
import re
import time
import hashlib
import logging
from typing import Optional
import requests

from config import (
    MISTRAL_API_KEY, MISTRAL_MODEL,
    MISTRAL_MAX_TOKENS, MISTRAL_TEMP
)

logger = logging.getLogger(__name__)

MISTRAL_URL = "https://api.mistral.ai/v1/chat/completions"
_CACHE: dict = {}
_TOKEN_LOG: dict = {"prompt": 0, "completion": 0, "calls": 0, "cache_hits": 0}


# ── [1] Text sanitisation ─────────────────────────────────────────────────────

def sanitize_for_prompt(text: str, max_chars: int = 1800) -> str:
    """
    Make PDF-extracted text safe to embed inside a prompt string.
    Problems we're fixing:
      • Raw "  in the text → breaks Mistral's JSON string output
      • Raw \n \t \r     → causes unterminated-string errors in JSON
      • Control chars    → confuse the tokeniser
      • Very long text   → pushes response past max_tokens
    """
    if not text:
        return ""

    # Replace smart quotes and other Unicode quote variants with plain apostrophe
    text = text.replace('\u201c', "'").replace('\u201d', "'")
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    text = text.replace('\u00ab', "'").replace('\u00bb', "'")

    # Replace all remaining double-quotes with single quotes
    # (safe: we're embedding as plain text, not as a JSON value)
    text = text.replace('"', "'")

    # Collapse whitespace: newlines/tabs → single space
    text = re.sub(r'[\r\n\t\f\v]+', ' ', text)
    text = re.sub(r' {2,}', ' ', text)

    # Strip control characters (ASCII 0–31 except space, and 127)
    text = re.sub(r'[\x00-\x1f\x7f]', '', text)

    # Strip backslashes (they escape nothing useful in plain-text context)
    text = text.replace('\\', ' ')

    return text[:max_chars].strip()


# ── [2] JSON repair ───────────────────────────────────────────────────────────

def _repair_json(raw: str) -> Optional[object]:
    """
    Multi-strategy JSON recovery for truncated / malformed responses.
    Strategy order (cheapest → most aggressive):
      1. Direct parse          — often works on first attempt
      2. Strip markdown fences — model sometimes wraps in ```json ... ```
      3. Extract first {...}   — grab the outermost object even if trailing garbage
      4. Extract first [...]   — same for arrays
      5. Truncation repair     — add missing closing brackets/braces
      6. Give up               — return None
    """
    if not raw:
        return None

    # 1. Direct
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # 2. Strip markdown
    cleaned = re.sub(r'^```(?:json)?\s*', '', raw.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r'\s*```$', '', cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # 3. Extract first JSON object { … }
    m = re.search(r'\{[\s\S]*\}', cleaned)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass

    # 4. Extract first JSON array [ … ]
    m = re.search(r'\[[\s\S]*\]', cleaned)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass

    # 5. Truncation repair — count open brackets and close them
    candidate = cleaned.strip()
    open_braces   = candidate.count('{') - candidate.count('}')
    open_brackets = candidate.count('[') - candidate.count(']')

    # Trim trailing incomplete key-value (e.g.  , "key": "unfinished)
    candidate = re.sub(r',\s*"[^"]*"?\s*:\s*"[^"]*$', '', candidate)
    candidate = re.sub(r',\s*"[^"]*"?\s*:\s*$',        '', candidate)
    candidate = re.sub(r',\s*$',                         '', candidate)

    # Re-count after trimming
    open_braces   = candidate.count('{') - candidate.count('}')
    open_brackets = candidate.count('[') - candidate.count(']')

    candidate += ']' * max(0, open_brackets)
    candidate += '}' * max(0, open_braces)

    try:
        parsed = json.loads(candidate)
        logger.debug("JSON repaired via truncation recovery.")
        return parsed
    except json.JSONDecodeError:
        pass

    logger.warning("JSON repair exhausted all strategies. Raw (first 120): %s", raw[:120])
    return None


# ── [3] Main call ─────────────────────────────────────────────────────────────

def call_mistral(
    prompt: str,
    system: str = "Return only valid JSON. No explanation.",
    max_tokens: int = MISTRAL_MAX_TOKENS,
    use_cache: bool = True,
    retries: int = 3,
) -> Optional[object]:
    """
    Call Mistral AI with retry + JSON repair.
    All prompts must have text sanitized via sanitize_for_prompt() before call.
    """
    key = hashlib.md5((system + prompt).encode()).hexdigest()
    if use_cache and key in _CACHE:
        logger.debug("Cache hit %s", key[:8])
        _TOKEN_LOG["cache_hits"] += 1
        return _CACHE[key]

    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type":  "application/json",
    }
    # json_object mode requires the model to return a JSON *object* (not bare array).
    # Prompts are written to return {"items":[...]} or {"result":{...}} so this is safe.
    body = {
        "model":           MISTRAL_MODEL,
        "max_tokens":      max_tokens,
        "temperature":     MISTRAL_TEMP,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
        ],
    }

    for attempt in range(retries):
        try:
            resp = requests.post(MISTRAL_URL, headers=headers, json=body, timeout=40)
            resp.raise_for_status()
            rj  = resp.json()
            raw = rj["choices"][0]["message"]["content"].strip()
            # Track real token usage from Mistral response
            usage = rj.get("usage", {})
            _TOKEN_LOG["prompt"]     += usage.get("prompt_tokens", 0)
            _TOKEN_LOG["completion"] += usage.get("completion_tokens", 0)
            _TOKEN_LOG["calls"]      += 1

            parsed = _repair_json(raw)
            if parsed is not None:
                if use_cache:
                    _CACHE[key] = parsed
                return parsed

            logger.warning("Mistral attempt %d: JSON repair failed.", attempt + 1)

        except requests.RequestException as e:
            logger.warning("Mistral attempt %d network error: %s", attempt + 1, e)
        except (KeyError, IndexError) as e:
            logger.warning("Mistral attempt %d bad response shape: %s", attempt + 1, e)

        if attempt < retries - 1:
            time.sleep(2 ** attempt)

    logger.error("All Mistral retries exhausted.")
    return None


def clear_cache():
    global _CACHE, _TOKEN_LOG
    _CACHE = {}
    _TOKEN_LOG = {"prompt": 0, "completion": 0, "calls": 0, "cache_hits": 0}


def get_token_usage() -> dict:
    """Return real Mistral token counts for this pipeline run."""
    total = _TOKEN_LOG["prompt"] + _TOKEN_LOG["completion"]
    return {
        "prompt_tokens":     _TOKEN_LOG["prompt"],
        "completion_tokens": _TOKEN_LOG["completion"],
        "total_tokens":      total,
        "api_calls":         _TOKEN_LOG["calls"],
        "cache_hits":        _TOKEN_LOG["cache_hits"],
    }


def reset_token_log():
    """Call before each pipeline run to get per-run counts."""
    global _TOKEN_LOG
    _TOKEN_LOG = {"prompt": 0, "completion": 0, "calls": 0, "cache_hits": 0}


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)
