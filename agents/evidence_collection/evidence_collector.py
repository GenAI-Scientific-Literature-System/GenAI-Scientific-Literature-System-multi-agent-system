# agents/evidence_collection/evidence_collector.py

import json
from typing import Iterable
from groq import Groq
from agents.evidence_collection.prompt import EVIDENCE_COLLECTION_PROMPT
from utils.api_keys import ApiKeyManager
from utils.model_config import DEFAULT_EVIDENCE_COLLECTION_MODEL, EVIDENCE_COLLECTION_FALLBACK_MODELS, ordered_models

class EvidenceCollector:
    def __init__(self, api_key: str | Iterable[str], model: str = DEFAULT_EVIDENCE_COLLECTION_MODEL):
        self.key_manager = ApiKeyManager.from_value(api_key)
        self.client = Groq(api_key=self.key_manager.current)
        self.model = model
        self.batch_size = 4  # papers per LLM call
        self.debug = True

    def _rotate_key(self):
        if not self.key_manager.rotate():
            return False
        self.client = Groq(api_key=self.key_manager.current)
        return True

    def _format_papers(self, papers: list[dict]) -> str:
        # papers is a list of {"paper_id": "...", "abstract": "..."}
        formatted = ""
        MAX_ABSTRACT_CHARS = 1500
        for p in papers:
            abstract = p.get("abstract", "")[:MAX_ABSTRACT_CHARS]
            formatted += f"[paper_id: {p['paper_id']}]\n{abstract}\n\n"
        return formatted.strip()

    def _call_llm(self, filled_prompt: str) -> str | None:
        last_error = None
        models_to_try = ordered_models(self.model, fallback_models=EVIDENCE_COLLECTION_FALLBACK_MODELS)

        for model in models_to_try:
            try:
                if self.debug:
                    print(f"[EvidenceCollector] Using key#{self.key_manager.position}, model: {model}")
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": filled_prompt}],
                    temperature=0.1  # lower than Agent 1 — classification needs to be consistent
                )
                raw = response.choices[0].message.content
                if not raw:
                    raise ValueError("Empty response from model")
                return raw.strip()

            except Exception as e:
                last_error = str(e)
                if "rate_limit" in str(e).lower() or "429" in str(e):
                    rotated = self._rotate_key()
                    if rotated:
                        try:
                            response = self.client.chat.completions.create(
                                model=model,
                                messages=[{"role": "user", "content": filled_prompt}],
                                temperature=0.1
                            )
                            raw = response.choices[0].message.content
                            if not raw:
                                raise ValueError("Empty response from model")
                            return raw.strip()
                        except Exception:
                            pass
                continue

        if self.debug:
            print(f"[EvidenceCollector] All models exhausted. Last error: {last_error}")
        return None

    def _parse_response(self, raw: str) -> list[dict] | None:
        # clean markdown wrapping
        if "```" in raw:
            parts = raw.split("```")
            if len(parts) >= 2:
                raw = parts[1]
            raw = raw.strip()
            if raw.startswith("json"):
                raw = raw[4:].strip()
        raw = raw.strip()

        if not raw:
            return None

        # attempt to extract JSON block if extra text exists
        if "{" in raw and "}" in raw:
            raw = raw[raw.find("{"):raw.rfind("}")+1]

        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            if self.debug:
                print(f"[EvidenceCollector] JSON parse failed. Raw: {raw}")
            return None

        # validate structure
        if "evidence" not in result:
            if self.debug:
                print("[EvidenceCollector] Missing 'evidence' key in response")
            return None

        valid = []
        for item in result["evidence"]:
            required_keys = ["paper_id", "classification", "reasoning"]
            missing = [k for k in required_keys if k not in item]
            if missing:
                if self.debug:
                    print(f"[EvidenceCollector] Skipping item, missing keys: {missing}")
                continue
            # enforce uppercase classification
            classification = item["classification"].strip().upper()
            if classification not in {"SUPPORTS", "CONTRADICTS", "INCONCLUSIVE"}:
                classification = "INCONCLUSIVE"
            item["classification"] = classification
            valid.append(item)

        return valid

    def collect(self, claim: str, papers: list[dict]) -> dict:
        # papers = [{"paper_id": "...", "abstract": "..."}, ...]
        all_evidence = []

        # split papers into batches
        for i in range(0, len(papers), self.batch_size):
            batch = papers[i: i + self.batch_size]
            if self.debug:
                print(f"[EvidenceCollector] Processing batch {i // self.batch_size + 1} ({len(batch)} papers)")

            formatted_papers = self._format_papers(batch)
            filled_prompt = EVIDENCE_COLLECTION_PROMPT.format(
                claim=claim,
                papers=formatted_papers
            )

            raw = self._call_llm(filled_prompt)
            if raw is None:
                # mark all papers in this batch as failed
                for p in batch:
                    all_evidence.append({
                        "paper_id": p["paper_id"],
                        "classification": "INCONCLUSIVE",
                        "reasoning": None,
                        "error": "LLM call failed"
                    })
                continue

            parsed = self._parse_response(raw)
            if parsed is None:
                for p in batch:
                    all_evidence.append({
                        "paper_id": p["paper_id"],
                        "classification": "INCONCLUSIVE",
                        "reasoning": None,
                        "error": "Parse failed"
                    })
                continue

            parsed_ids = {e["paper_id"] for e in parsed}
            for p in batch:
                if p["paper_id"] not in parsed_ids:
                    parsed.append({
                        "paper_id": p["paper_id"],
                        "classification": "INCONCLUSIVE",
                        "reasoning": None,
                        "error": "Missing in LLM output"
                    })

            all_evidence.extend(parsed)

        seen = set()
        deduped = []
        for e in all_evidence:
            if e["paper_id"] not in seen:
                deduped.append(e)
                seen.add(e["paper_id"])
        all_evidence = deduped

        # group into supporting, contradicting, inconclusive
        result = {
            "claim": claim,
            "supporting": [e for e in all_evidence if e["classification"] == "SUPPORTS"],
            "contradicting": [e for e in all_evidence if e["classification"] == "CONTRADICTS"],
            "inconclusive": [e for e in all_evidence if e["classification"] == "INCONCLUSIVE"]
        }

        return result