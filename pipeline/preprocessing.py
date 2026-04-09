# pipeline/preprocessing.py

import re
import json
from groq import Groq
from utils.model_config import ordered_models, DEFAULT_CLAIM_EXTRACTION_MODEL
from utils.api_keys import ApiKeyManager

DOMAIN_CLASSIFICATION_PROMPT = """
You are a research domain classification agent.

Given a user's research query, classify it into AT MOST 2 domains from the list below.

Domains:
- medical       (clinical studies, drugs, diseases, patient outcomes)
- biology       (molecular biology, genetics, biochemistry, neuroscience)
- ml            (machine learning, deep learning, AI models, neural networks)
- general       (interdisciplinary, none of the above clearly apply)

Rules:
- Return AT MOST 2 domains
- Only return domains that clearly apply
- If unsure, prefer "general"
- Do not explain your answer

Return ONLY valid JSON. No text outside the JSON.

Query:
{query}

Output format:
{{
  "domains": ["<domain1>", "<domain2>"]
}}
"""

VALID_DOMAINS = {"medical", "biology", "ml", "general"}


class QueryPreprocessor:
    def __init__(self, api_key: str | list[str], model: str = DEFAULT_CLAIM_EXTRACTION_MODEL, debug: bool = True):
        self.key_manager = ApiKeyManager.from_value(api_key)
        self.models = ordered_models(model)
        self.client = Groq(api_key=self.key_manager.current)
        self.debug = debug

    def clean(self, query: str) -> str:
        query = query.strip()
        query = re.sub(r"\s+", " ", query)          # collapse whitespace
        query = re.sub(r"[^\w\s\-\?\.,:%/]", "", query) # strip special chars
        return query

    def classify_domain(self, query: str) -> list[str]:
        filled_prompt = DOMAIN_CLASSIFICATION_PROMPT.format(query=query)

        last_error = None
        for model in self.models:
            try:
                if self.debug:
                    print(f"[QueryPreprocessor] Using model: {model}")
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": filled_prompt}],
                    temperature=0.1
                )
                raw = response.choices[0].message.content.strip()
                break

            except Exception as e:
                last_error = str(e)
                if "rate_limit" in str(e).lower() or "429" in str(e):
                    if self.key_manager.rotate():
                        self.client = Groq(api_key=self.key_manager.current)
                        try:
                            response = self.client.chat.completions.create(
                                model=model,
                                messages=[{"role": "user", "content": filled_prompt}],
                                temperature=0.1
                            )
                            raw = response.choices[0].message.content.strip()
                            break
                        except Exception:
                            pass
                continue
        else:
            if self.debug:
                print(f"[QueryPreprocessor] All models failed: {last_error}")
            return ["general"]  # safe default

        if not raw:
            return ["general"]

        # parse
        if "```" in raw:
            parts = raw.split("```")
            if len(parts) >= 2:
                raw = parts[1].strip()
            if raw.startswith("json"):
                raw = raw[4:].strip()
        raw = raw.strip()

        # extract JSON block if extra text exists
        if "{" in raw and "}" in raw:
            raw = raw[raw.find("{"):raw.rfind("}")+1]

        try:
            result = json.loads(raw)
            domains = result.get("domains", [])
        except json.JSONDecodeError:
            if self.debug:
                print(f"[QueryPreprocessor] JSON parse failed, defaulting to general")
            return ["general"]

        # validate + filter
        cleaned_domains = []
        for d in domains:
            d_clean = d.lower().strip() if isinstance(d, str) else str(d).lower().strip()
            if d_clean in VALID_DOMAINS:
                cleaned_domains.append(d_clean)
        
        cleaned_domains = list(dict.fromkeys(cleaned_domains))

        if "general" in cleaned_domains and len(cleaned_domains) > 1:
            cleaned_domains.remove("general")

        domains = cleaned_domains[:2]

        if not domains:
            domains = ["general"]

        return domains

    def process(self, query: str) -> dict:
        cleaned = self.clean(query)
        domains = self.classify_domain(cleaned)
        return {
            "original_query": query,
            "cleaned_query": cleaned,
            "domains": domains
        }