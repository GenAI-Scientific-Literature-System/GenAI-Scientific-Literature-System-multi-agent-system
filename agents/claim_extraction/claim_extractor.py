# agents/claim_extraction/claim_extractor.py

import json
from typing import Iterable

from groq import Groq
from agents.claim_extraction.prompt import CLAIM_EXTRACTION_PROMPT
from utils.api_keys import ApiKeyManager
from utils.model_config import DEFAULT_CLAIM_EXTRACTION_MODEL, ordered_models

class ClaimExtractor:
    def __init__(self, api_key: str | Iterable[str], model: str = DEFAULT_CLAIM_EXTRACTION_MODEL):
        self.key_manager = ApiKeyManager.from_value(api_key)
        self.client = Groq(api_key=self.key_manager.current)
        self.model = model

    def _rotate_key(self):
        if not self.key_manager.rotate():
            return False  # no more keys
        self.client = Groq(api_key=self.key_manager.current)
        return True
    
    def extract(self, paper_text: str) -> dict:
        MAX_CHARS = 6000
        if len(paper_text) > MAX_CHARS:
            paper_text = paper_text[:MAX_CHARS]
            last_period = paper_text.rfind(".")
            if last_period != -1:
                paper_text = paper_text[:last_period + 1]

        filled_prompt = CLAIM_EXTRACTION_PROMPT.format(paper_text=paper_text)

        last_error = None
        models_to_try = ordered_models(self.model)
        for model in models_to_try:
            try:
                print(f"[ClaimExtractor] Using key#{self.key_manager.position}, model: {model}")
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": filled_prompt}],
                    temperature=0.2
                )
                raw = response.choices[0].message.content.strip()
                break

            except Exception as e:
                last_error = str(e)
                err = str(e).lower()
                # rotate key on rate limit or invalid/unauthorized key errors
                if (
                    "rate_limit" in err
                    or "429" in err
                    or "invalid_api_key" in err
                    or "invalid api key" in err
                    or "401" in err
                    or "unauthorized" in err
                ):
                    rotated = self._rotate_key()
                    if rotated:
                        try:
                            print(f"[ClaimExtractor] Retrying with key#{self.key_manager.position}, model: {model}")
                            response = self.client.chat.completions.create(
                                model=model,
                                messages=[{"role": "user", "content": filled_prompt}],
                                temperature=0.2
                            )
                            raw = response.choices[0].message.content.strip()
                            break
                        except Exception:
                            pass
                continue
        else:
            return {
                "claim": None,
                "confidence": None,
                "reasoning": None,
                "error": f"All models and keys exhausted. Last error: {last_error}"
            }

        # clean garbage wrapping before parsing
        if "```" in raw:
            parts = raw.split("```")
            if len(parts) >= 2:
                raw = parts[1]
            raw = raw.strip()  # strip first
            if raw.startswith("json"):
                raw = raw[4:].strip()
        raw = raw.strip()

        if not raw:
            return {
                "claim": None,
                "confidence": None,
                "reasoning": None,
                "error": "Empty response from model"
            }

        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            return {
                "subject": None,
                "predicate": None,
                "object": None,
                "claim": None,
                "confidence": None,
                "reasoning": None,
                "error": f"Failed to parse JSON. Raw response: {raw}"
            }

        # validate keys
        required_keys = ["subject", "predicate", "object", "claim", "confidence", "reasoning"]
        missing = [k for k in required_keys if k not in result]
        if missing:
            result["error"] = f"Missing required fields: {missing}"

        subject = (result.get("subject") or "").strip()
        predicate = (result.get("predicate") or "").strip()
        obj = (result.get("object") or "").strip()
        claim_text = (result.get("claim") or "").strip()

        # If claim text is missing but SPO is present, synthesize claim text.
        if not claim_text and (subject or predicate or obj):
            claim_text = " ".join(p for p in [subject, predicate, obj] if p).strip()

        # Normalize final payload for downstream consumers.
        result["subject"] = subject or None
        result["predicate"] = predicate or None
        result["object"] = obj or None
        result["claim"] = claim_text or None

        return result