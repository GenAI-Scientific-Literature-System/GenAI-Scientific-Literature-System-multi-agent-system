"""
MERLIN LLM Client
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
    GROQ_MODEL,
    GROQ_MAX_TOKENS, GROQ_TEMP, GROQ_API_KEYS
)

logger = logging.getLogger(__name__)

LLM_URL = "https://api.groq.com/openai/v1/chat/completions"
_CACHE: dict = {}
_TOKEN_LOG: dict = {"prompt": 0, "completion": 0, "calls": 0, "cache_hits": 0}


# ── [1] Text sanitisation ─────────────────────────────────────────────────────

def sanitize_for_prompt(text: str, max_chars: int = 1800) -> str:
    """
    Make PDF-extracted text safe to embed inside a prompt string.
    Problems we're fixing:
      • Raw "  in the text → breaks LLM's JSON string output
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

def call_llm(
    prompt: str,
    system: str = "Return only valid JSON. No explanation.",
    max_tokens: int = GROQ_MAX_TOKENS,
    use_cache: bool = True,
    retries: int = 3,
) -> Optional[object]:
    """
    Call LLM with API key rotation per model + JSON repair.
    """
    key = hashlib.md5((system + prompt).encode()).hexdigest()
    if use_cache and key in _CACHE:
        logger.debug("Cache hit %s", key[:8])
        _TOKEN_LOG["cache_hits"] += 1
        return _CACHE[key]

    fallback_models = [
        GROQ_MODEL,
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768"
    ]
    
    if not GROQ_API_KEYS:
        logger.error("No Groq API keys found.")
        return None

    # We try each model
    for current_model in fallback_models:
        body = {
            "model":           current_model,
            "max_tokens":      max_tokens,
            "temperature":     GROQ_TEMP,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": prompt},
            ],
        }

        # For every model we try every api key until it works
        model_success = False
        import random
        keys_to_try = GROQ_API_KEYS.copy()
        random.shuffle(keys_to_try) # Optional: distribute load randomly
        
        for idx, api_key in enumerate(keys_to_try):
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type":  "application/json",
            }
            
            for attempt in range(retries):
                try:
                    resp = requests.post(LLM_URL, headers=headers, json=body, timeout=40)
                    
                    # 429 = Ratelimit, switch to next key without more retries on this key for this request
                    if resp.status_code == 429:
                        logger.warning("Agent hit 429 on %s for key #%d. Trying next key.", current_model, idx+1)
                        # Break out of the 'attempt' loop to immediately rotate to next API key
                        break
                        
                    resp.raise_for_status()
                    rj  = resp.json()
                    raw = rj["choices"][0]["message"]["content"].strip()
                    
                    # Track real token usage
                    usage = rj.get("usage", {})
                    _TOKEN_LOG["prompt"]     += usage.get("prompt_tokens", 0)
                    _TOKEN_LOG["completion"] += usage.get("completion_tokens", 0)
                    _TOKEN_LOG["calls"]      += 1

                    parsed = _repair_json(raw)
                    if parsed is not None:
                        if use_cache:
                            _CACHE[key] = parsed
                        return parsed

                    logger.warning("LLM JSON repair failed on model %s.", current_model)

                except requests.RequestException as e:
                    logger.warning("Network error on %s: %s", current_model, e)
                except (KeyError, IndexError) as e:
                    logger.warning("Bad response shape on %s: %s", current_model, e)

                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
            else:
                # If the 'attempt' loop didn't break out (meaning no 429), but failed continuously 
                # (e.g. timeout, JSON schema failure), we also try the next key
                pass
                
        # If we exhausted ALL keys for this model without returning a parsed object,
        # we fall back to the NEXT model in the list.
        logger.warning("Exhausted all API keys for model: %s. Falling back to next model.", current_model)

    logger.error("All retries, API Keys, and fallback models exhausted.")
    return None


def clear_cache():
    global _CACHE, _TOKEN_LOG
    _CACHE = {}
    _TOKEN_LOG = {"prompt": 0, "completion": 0, "calls": 0, "cache_hits": 0}


def get_token_usage() -> dict:
    """Return real LLM token counts for this pipeline run."""
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
