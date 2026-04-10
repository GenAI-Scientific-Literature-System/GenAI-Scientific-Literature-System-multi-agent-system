"""
MERLIN AI Client  —  Mistral primary, Groq fallback
Fixes:
  [1] sanitize_for_prompt() — strips chars that break JSON output
  [2] _repair_json()        — recovers partial/truncated responses
  [3] max_tokens raised     — prevents mid-string cutoff
  [4] array wrapping        — json_object mode needs an object not bare array
  [5] Groq fallback         — if Mistral returns 401/exhausted, retry via Groq
"""
import json
import os
import re
import time
import hashlib
import logging
from typing import Optional
import requests

from config import (
    MISTRAL_API_KEY, MISTRAL_MODEL,
    MISTRAL_MAX_TOKENS, MISTRAL_TEMP,
    GROQ_API_KEY, GROQ_MODEL,
)

logger = logging.getLogger(__name__)

MISTRAL_URL = "https://api.mistral.ai/v1/chat/completions"
GROQ_URL    = "https://api.groq.com/openai/v1/chat/completions"

_CACHE: dict = {}
_TOKEN_LOG: dict = {"prompt": 0, "completion": 0, "calls": 0, "cache_hits": 0}


# ── [1] Text sanitisation ─────────────────────────────────────────────────────

def sanitize_for_prompt(text: str, max_chars: int = 1800) -> str:
    if not text:
        return ""
    text = text.replace('\u201c', "'").replace('\u201d', "'")
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    text = text.replace('\u00ab', "'").replace('\u00bb', "'")
    text = text.replace('"', "'")
    text = re.sub(r'[\r\n\t\f\v]+', ' ', text)
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'[\x00-\x1f\x7f]', '', text)
    text = text.replace('\\', ' ')
    return text[:max_chars].strip()


# ── [2] JSON repair ───────────────────────────────────────────────────────────

def _repair_json(raw: str) -> Optional[object]:
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    cleaned = re.sub(r'^```(?:json)?\s*', '', raw.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r'\s*```$', '', cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    m = re.search(r'\{[\s\S]*\}', cleaned)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    m = re.search(r'\[[\s\S]*\]', cleaned)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    candidate = cleaned.strip()
    candidate = re.sub(r',\s*"[^"]*"?\s*:\s*"[^"]*$', '', candidate)
    candidate = re.sub(r',\s*"[^"]*"?\s*:\s*$',        '', candidate)
    candidate = re.sub(r',\s*$',                         '', candidate)
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


# ── [3] Groq call (OpenAI-compatible endpoint) ───────────────────────────────

def _call_groq(
    prompt: str,
    system: str,
    max_tokens: int,
    retries: int = 3,
) -> Optional[object]:
    """Call Groq as a fallback using its OpenAI-compatible endpoint."""
    api_key = str(GROQ_API_KEY)
    if not api_key:
        for i in range(1, 7):
            api_key = os.environ.get(f"GROQ_API_KEY_{i}", "")
            if api_key:
                break
    if not api_key:
        logger.error("Groq fallback: no GROQ_API_KEY found in environment.")
        return None

    model = str(GROQ_MODEL) or "llama-3.3-70b-versatile"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
    }
    body = {
        "model":           model,
        "max_tokens":      max_tokens,
        "temperature":     float(MISTRAL_TEMP),
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
        ],
    }

    for attempt in range(retries):
        try:
            resp = requests.post(GROQ_URL, headers=headers, json=body, timeout=40)
            # Some Groq models don't support json_object — fall back to plain text
            if resp.status_code == 400 and "response_format" in body:
                logger.warning("Groq attempt %d: 400 with json_object — retrying without.", attempt + 1)
                body.pop("response_format", None)
                resp = requests.post(GROQ_URL, headers=headers, json=body, timeout=40)
            resp.raise_for_status()
            rj  = resp.json()
            raw = rj["choices"][0]["message"]["content"].strip()
            usage = rj.get("usage", {})
            _TOKEN_LOG["prompt"]     += usage.get("prompt_tokens", 0)
            _TOKEN_LOG["completion"] += usage.get("completion_tokens", 0)
            _TOKEN_LOG["calls"]      += 1
            parsed = _repair_json(raw)
            if parsed is not None:
                logger.info("Groq fallback succeeded (attempt %d, model=%s).", attempt + 1, model)
                return parsed
            logger.warning("Groq attempt %d: JSON repair failed.", attempt + 1)
        except requests.RequestException as e:
            logger.warning("Groq attempt %d network error: %s", attempt + 1, e)
        except (KeyError, IndexError) as e:
            logger.warning("Groq attempt %d bad response shape: %s", attempt + 1, e)
        if attempt < retries - 1:
            time.sleep(2 ** attempt)

    logger.error("Groq fallback: all retries exhausted.")
    return None


# ── [4] Main call — Mistral primary, Groq fallback ───────────────────────────

def call_mistral(
    prompt: str,
    system: str = "Return only valid JSON. No explanation.",
    max_tokens: int = MISTRAL_MAX_TOKENS,
    use_cache: bool = True,
    retries: int = 3,
) -> Optional[object]:
    """
    Call Mistral AI with retry + JSON repair.
    On 401 / total exhaustion, transparently falls back to Groq.
    """
    key = hashlib.md5((system + prompt).encode()).hexdigest()
    if use_cache and key in _CACHE:
        logger.debug("Cache hit %s", key[:8])
        _TOKEN_LOG["cache_hits"] += 1
        return _CACHE[key]

    mistral_key = str(MISTRAL_API_KEY)

    if mistral_key:
        headers = {
            "Authorization": f"Bearer {mistral_key}",
            "Content-Type":  "application/json",
        }
        body = {
            "model":           str(MISTRAL_MODEL),
            "max_tokens":      int(max_tokens),
            "temperature":     float(MISTRAL_TEMP),
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": prompt},
            ],
        }

        mistral_dead = False
        for attempt in range(retries):
            try:
                resp = requests.post(MISTRAL_URL, headers=headers, json=body, timeout=40)
                # 401 = bad key — abort immediately, no more Mistral retries
                if resp.status_code == 401:
                    logger.warning(
                        "Mistral 401 Unauthorized — key invalid/expired. Switching to Groq."
                    )
                    mistral_dead = True
                    break
                resp.raise_for_status()
                rj  = resp.json()
                raw = rj["choices"][0]["message"]["content"].strip()
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

        if not mistral_dead:
            logger.error("All Mistral retries exhausted — switching to Groq fallback.")
    else:
        logger.info("No MISTRAL_API_KEY configured — using Groq directly.")

    # ── Groq fallback ─────────────────────────────────────────────────────────
    result = _call_groq(prompt, system, int(max_tokens), retries=retries)
    if result is not None and use_cache:
        _CACHE[key] = result
    return result


def clear_cache():
    global _CACHE, _TOKEN_LOG
    _CACHE = {}
    _TOKEN_LOG = {"prompt": 0, "completion": 0, "calls": 0, "cache_hits": 0}


def get_token_usage() -> dict:
    total = _TOKEN_LOG["prompt"] + _TOKEN_LOG["completion"]
    return {
        "prompt_tokens":     _TOKEN_LOG["prompt"],
        "completion_tokens": _TOKEN_LOG["completion"],
        "total_tokens":      total,
        "api_calls":         _TOKEN_LOG["calls"],
        "cache_hits":        _TOKEN_LOG["cache_hits"],
    }


def reset_token_log():
    global _TOKEN_LOG
    _TOKEN_LOG = {"prompt": 0, "completion": 0, "calls": 0, "cache_hits": 0}


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)
