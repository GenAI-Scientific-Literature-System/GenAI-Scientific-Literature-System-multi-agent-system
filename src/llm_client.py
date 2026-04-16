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
import ast
from typing import Optional
import requests

from config import (
    GROQ_MODEL,
    GROQ_MAX_TOKENS, GROQ_TEMP, GROQ_API_KEYS, GROQ_FALLBACK_MODELS
)

logger = logging.getLogger(__name__)

LLM_URL = "https://api.groq.com/openai/v1/chat/completions"
_CACHE: dict = {}
_TOKEN_LOG: dict = {"prompt": 0, "completion": 0, "calls": 0, "cache_hits": 0}
_MODEL_COOLDOWN_UNTIL: dict[str, float] = {}
_RATE_LIMIT_COOLDOWN_SEC = 45


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

    def _try_json(text: str) -> Optional[object]:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    def _extract_balanced_candidates(text: str) -> list[str]:
        """Extract balanced {...} / [...] blocks while respecting strings."""
        candidates: list[str] = []
        for opener, closer in (('{', '}'), ('[', ']')):
            start = text.find(opener)
            while start != -1:
                depth = 0
                in_string = False
                escape = False
                for i in range(start, len(text)):
                    ch = text[i]
                    if in_string:
                        if escape:
                            escape = False
                        elif ch == '\\':
                            escape = True
                        elif ch == '"':
                            in_string = False
                        continue

                    if ch == '"':
                        in_string = True
                    elif ch == opener:
                        depth += 1
                    elif ch == closer:
                        depth -= 1
                        if depth == 0:
                            candidates.append(text[start:i + 1])
                            break
                start = text.find(opener, start + 1)
        return candidates

    def _normalize_loose_json(text: str) -> str:
        t = text.strip()

        # Remove prose wrappers while keeping first likely JSON block.
        candidates = _extract_balanced_candidates(t)
        if candidates:
            t = max(candidates, key=len)

        # Common JSON-like fixes.
        t = t.replace('\u201c', '"').replace('\u201d', '"').replace('\u2018', "'").replace('\u2019', "'")
        t = re.sub(r',\s*([}\]])', r'\1', t)  # trailing commas
        t = re.sub(r'\bTrue\b', 'true', t)
        t = re.sub(r'\bFalse\b', 'false', t)
        t = re.sub(r'\bNone\b', 'null', t)

        # Quote bare keys: {foo: 1} -> {"foo": 1}
        t = re.sub(r'([\{,]\s*)([A-Za-z_][A-Za-z0-9_\-\s]*)(\s*:)', r'\1"\2"\3', t)

        # Convert single-quoted strings to double-quoted strings.
        t = re.sub(
            r"'([^'\\]*(?:\\.[^'\\]*)*)'",
            lambda m: '"' + m.group(1).replace('"', '\\"') + '"',
            t,
        )
        return t

    # 1. Direct
    parsed = _try_json(raw)
    if parsed is not None:
        return parsed

    # 2. Strip markdown
    cleaned = re.sub(r'^```(?:json)?\s*', '', raw.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r'\s*```$', '', cleaned)
    parsed = _try_json(cleaned)
    if parsed is not None:
        return parsed

    # 3. Try balanced JSON candidates first (safer than greedy regex).
    for candidate in sorted(_extract_balanced_candidates(cleaned), key=len, reverse=True):
        parsed = _try_json(candidate)
        if parsed is not None:
            return parsed

    # 4. Try normalized loose-JSON repair.
    normalized = _normalize_loose_json(cleaned)
    parsed = _try_json(normalized)
    if parsed is not None:
        logger.debug("JSON repaired via loose-json normalization.")
        return parsed

    # 5. Python-literal fallback for dict/list-like outputs.
    try:
        literal = ast.literal_eval(normalized)
        if isinstance(literal, (dict, list)):
            logger.debug("JSON repaired via literal_eval fallback.")
            return literal
    except Exception:
        pass

    # 6. Truncation repair — count open brackets and close them
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

    parsed = _try_json(candidate)
    if parsed is not None:
        logger.debug("JSON repaired via truncation recovery.")
        return parsed

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

    expects_json = "json" in (system or "").lower()
    # OSS model is useful for free-form text, but unreliable for strict structured JSON extraction.
    fallback_models = [
        m for m in GROQ_FALLBACK_MODELS
        if (not expects_json) or ("gpt-oss-120b" not in m.lower())
    ]
    
    if not GROQ_API_KEYS:
        logger.error("No Groq API keys found.")
        return None

    logger.info(
        "LLM request model chain (%s): %s",
        "json" if expects_json else "freeform",
        " -> ".join(fallback_models),
    )

    # We try each model
    for current_model in fallback_models:
        cooldown_until = _MODEL_COOLDOWN_UNTIL.get(current_model, 0.0)
        now = time.time()
        if cooldown_until > now:
            logger.info(
                "Skipping model %s for %.1fs due to recent rate-limit cooldown.",
                current_model,
                cooldown_until - now,
            )
            continue

        logger.info("Attempting model: %s", current_model)
        model_max_tokens = max_tokens
        enforce_json_object = True
        force_next_model = False
        rate_limited_keys = 0

        # For every model we try every api key until it works
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
                    body = {
                        "model":       current_model,
                        "max_tokens":  model_max_tokens,
                        "temperature": GROQ_TEMP,
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user",   "content": prompt},
                        ],
                    }
                    if enforce_json_object:
                        body["response_format"] = {"type": "json_object"}

                    resp = requests.post(LLM_URL, headers=headers, json=body, timeout=40)
                    
                    # 429 = Ratelimit, switch to next key without more retries on this key for this request
                    if resp.status_code == 429:
                        logger.warning("Agent hit 429 on %s for key #%d. Trying next key.", current_model, idx+1)
                        rate_limited_keys += 1
                        # Break out of the 'attempt' loop to immediately rotate to next API key
                        break

                    # 400 = payload/model mismatch. Adapt request and retry.
                    if resp.status_code == 400:
                        try:
                            err = resp.json()
                            err_msg = (
                                err.get("error", {}).get("message")
                                if isinstance(err, dict)
                                else str(err)
                            ) or resp.text
                        except Exception:
                            err_msg = resp.text

                        logger.warning(
                            "Bad request on %s (key #%d): %s",
                            current_model,
                            idx + 1,
                            str(err_msg)[:280],
                        )

                        lowered = str(err_msg).lower()
                        adapted = False

                        if enforce_json_object and (
                            "response_format" in lowered
                            or "json_object" in lowered
                            or "failed to validate json" in lowered
                            or "failed_generation" in lowered
                            or "unsupported" in lowered
                            or "not supported" in lowered
                        ):
                            enforce_json_object = False
                            adapted = True
                            logger.info("Retrying %s without response_format json_object.", current_model)

                        if ("max_tokens" in lowered or "token" in lowered) and model_max_tokens > 2048:
                            prev = model_max_tokens
                            model_max_tokens = max(2048, min(model_max_tokens, 4096))
                            adapted = adapted or (model_max_tokens != prev)
                            if model_max_tokens != prev:
                                logger.info(
                                    "Retrying %s with reduced max_tokens=%d (was %d).",
                                    current_model,
                                    model_max_tokens,
                                    prev,
                                )

                        if adapted:
                            continue

                        # Unrecoverable bad request for this key; try next key.
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

                    # If strict JSON mode was already disabled and the output is still
                    # not parseable, this model is not suitable for this structured call.
                    # Move to next fallback model instead of burning more keys/retries.
                    if not enforce_json_object:
                        logger.warning(
                            "Model %s returned non-JSON output after relaxed mode; "
                            "switching to next fallback model.",
                            current_model,
                        )
                        force_next_model = True
                        break

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

            if force_next_model:
                break
                
        # If we exhausted ALL keys for this model without returning a parsed object,
        # we fall back to the NEXT model in the list.
        if rate_limited_keys >= len(keys_to_try) and len(keys_to_try) > 0:
            _MODEL_COOLDOWN_UNTIL[current_model] = time.time() + _RATE_LIMIT_COOLDOWN_SEC
            logger.warning(
                "Model %s placed on %.0fs cooldown after full 429 sweep.",
                current_model,
                float(_RATE_LIMIT_COOLDOWN_SEC),
            )

        if force_next_model:
            logger.warning("Model %s unsuitable for structured JSON output. Falling back to next model.", current_model)
        else:
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
