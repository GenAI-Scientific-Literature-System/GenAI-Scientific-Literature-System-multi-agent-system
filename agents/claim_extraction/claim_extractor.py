# agents/claim_extraction/claim_extractor.py

import json
from groq import Groq
from agents.claim_extraction.prompt import CLAIM_EXTRACTION_PROMPT

class ClaimExtractor:
    def __init__(self, api_key: str | list[str], model: str = "llama-3.3-70b-versatile"):
        self.api_keys = [api_key] if isinstance(api_key, str) else api_key
        self.current_key_index = 0
        self.client = Groq(api_key=self.api_keys[0])
        self.model = model
        self.fallback_models = [
            "llama-3.3-70b-versatile",                # Meta, latest 70b
            "openai/gpt-oss-120b",                    # OpenAI OSS, 120b
            "moonshotai/kimi-k2-instruct-0905",       # Moonshot, 256K context
            "moonshotai/kimi-k2-instruct",            # Moonshot, alternate
            "meta-llama/llama-4-scout-17b-16e-instruct",  # Meta Llama 4 MoE
            "qwen/qwen3-32b",                         # Alibaba, 32b reasoning
            "openai/gpt-oss-20b",                     # OpenAI OSS, 20b
            "llama-3.1-8b-instant",                   # Meta, 8b fast fallback
        ]

    def _rotate_key(self):
        self.current_key_index += 1
        if self.current_key_index >= len(self.api_keys):
            return False  # no more keys
        self.client = Groq(api_key=self.api_keys[self.current_key_index])
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
        models_to_try = [self.model] + [m for m in self.fallback_models if m != self.model]
        for model in models_to_try:
            try:
                print(f"[ClaimExtractor] Using key#{self.current_key_index + 1}, model: {model}")
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
                            print(f"[ClaimExtractor] Retrying with key#{self.current_key_index + 1}, model: {model}")
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
                "claim": None,
                "confidence": None,
                "reasoning": None,
                "error": f"Failed to parse JSON. Raw response: {raw}"
            }

        # validate keys
        required_keys = ["claim", "confidence", "reasoning"]
        missing = [k for k in required_keys if k not in result]
        if missing:
            result["error"] = f"Missing required fields: {missing}"

        return result