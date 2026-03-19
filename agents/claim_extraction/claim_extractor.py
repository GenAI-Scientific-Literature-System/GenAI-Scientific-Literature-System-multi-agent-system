# agents/claim_extraction/claim_extractor.py

import json
from groq import Groq
from agents.claim_extraction.prompt import CLAIM_EXTRACTION_PROMPT

class ClaimExtractor:
    def __init__(self, api_key: str | list[str], model: str = "llama3-70b-8192"):
        self.api_keys = [api_key] if isinstance(api_key, str) else api_key
        self.current_key_index = 0
        self.client = Groq(api_key=self.api_keys[0])
        self.model = model
        self.fallback_models = [
            "llama3-70b-8192",                # Meta, strong general purpose
            "openai/gpt-oss-120b",            # ChatGPT OSS fallback
            "llama-3.1-70b-versatile",        # Meta, improved instruction following
            "llama-3.3-70b-versatile",        # Meta, latest 70b
            "deepseek-r1-distill-llama-70b",  # DeepSeek, strong reasoning
            "qwen-qwq-32b",                   # Alibaba, strong reasoning
            "mixtral-8x7b-32768",             # Mistral, long context
            "gemma2-9b-it",                   # Google, lightweight fallback
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
                print(f"[ClaimExtractor] Using model: {model}")
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": filled_prompt}],
                    temperature=0.2
                )
                raw = response.choices[0].message.content.strip()
                break

            except Exception as e:
                last_error = str(e)
                # if rate limited, try rotating API key before next model
                if "rate_limit" in str(e).lower() or "429" in str(e):
                    rotated = self._rotate_key()
                    if rotated:
                        try:
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