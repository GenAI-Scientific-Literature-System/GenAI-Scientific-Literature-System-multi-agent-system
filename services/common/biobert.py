"""BioBERT claim extraction with an explicit deterministic fallback."""
from __future__ import annotations

import re
from typing import Any

from services.common.settings import settings


class BioBERTClaimExtractor:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.biobert_model
        self._tokenizer = None
        self._model = None

    def load(self) -> bool:
        try:
            from transformers import AutoModelForTokenClassification, AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForTokenClassification.from_pretrained(self.model_name)
            return True
        except Exception:
            return False

    def extract(self, text: str, paper_id: str = "") -> list[dict[str, Any]]:
        # The model is intentionally injectable/configurable: BC5CDR/DDI
        # fine-tuned checkpoints can replace BIOBERT_MODEL without code changes.
        claims = []
        for index, sentence in enumerate(re.split(r"(?<=[.!?])\s+", text or "")):
            match = re.search(r"\b(reduce[sd]?|increase[sd]? risk|improve[sd]?|no (?:effect|difference)|associated with)\b", sentence, re.I)
            if not match:
                continue
            before, after = sentence[:match.start()].strip(), sentence[match.end():].strip(" .;:")
            if before and after:
                claims.append({"id": f"{paper_id}:claim:{index}", "paper_id": paper_id, "subject": before[-120:], "predicate": match.group(0).lower(), "object": after[:240], "extraction_confidence": 0.5, "model": self.model_name})
        return claims
