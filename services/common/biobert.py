"""BioBERT token classification and biomedical NER inference engine.

Provides BC5CDR / DDI biomedical entity recognition (Chemicals, Diseases, Drugs, Outcomes)
using fine-tuned transformer token classification checkpoints (d4data/biomedical-ner-all,
alvaroalon2/biobert_chemical_ner), with a deterministic clinical heuristic fallback.
"""
from __future__ import annotations

import logging
import re
from typing import Any, List, Dict, Tuple

from services.common.settings import settings

logger = logging.getLogger(__name__)


class BioBERTClaimExtractor:
    def __init__(self, model_name: str | None = None, auto_load: bool = True):
        self.model_name = model_name or settings.biobert_model
        self._tokenizer = None
        self._model = None
        self._torch = None
        self._is_loaded = False
        self._load_attempted = False
        if auto_load:
            self.load()

    def load(self) -> bool:
        """Load pretrained BioBERT / SciBERT token classification weights."""
        if self._is_loaded:
            return True
        self._load_attempted = True
        try:
            import os
            os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
            os.environ["OMP_NUM_THREADS"] = "1"
            import torch
            from transformers import AutoModelForTokenClassification, AutoTokenizer
            self._torch = torch
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForTokenClassification.from_pretrained(self.model_name)
            self._model.eval()
            self._is_loaded = True
            logger.info("Successfully loaded BioBERT NER checkpoint: %s", self.model_name)
            return True
        except Exception as e:
            logger.debug("BioBERT transformer loading skipped: %s", e)
            self._is_loaded = False
            return False

    def _infer_entities(self, sentence: str) -> Tuple[List[Dict[str, Any]], float]:
        """Perform tensor-level token classification inference."""
        if not self._is_loaded and not self._load_attempted:
            self.load()

        if not self._is_loaded or self._model is None or self._tokenizer is None:
            return [], 0.5

        try:
            inputs = self._tokenizer(sentence, return_tensors="pt", truncation=True, max_length=512)
            with self._torch.no_grad():
                outputs = self._model(**inputs)
                logits = outputs.logits
                probs = self._torch.softmax(logits, dim=-1)
                conf = float(self._torch.max(probs).item())
                predictions = self._torch.argmax(logits, dim=2)
            
            tokens = self._tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            entities = []
            current_entity = []
            current_label = None

            id2label = getattr(self._model.config, "id2label", {})
            for token, pred_id in zip(tokens, predictions[0].tolist()):
                if token in [self._tokenizer.cls_token, self._tokenizer.sep_token, self._tokenizer.pad_token]:
                    continue
                label = id2label.get(pred_id, "O")
                if label != "O":
                    if token.startswith("##"):
                        if current_entity:
                            current_entity[-1] += token[2:]
                    else:
                        current_entity.append(token)
                        current_label = label
                else:
                    if current_entity and current_label:
                        raw_ent = " ".join(current_entity)
                        clean_ent = re.sub(r'\s+([,.:;?!\-\(\)/])', r'\1', raw_ent)
                        clean_ent = re.sub(r'([,.:;?!\-\(\)/])\s+', r'\1', clean_ent).strip()
                        entities.append({"text": clean_ent, "label": current_label})
                        current_entity = []
                        current_label = None
            if current_entity and current_label:
                raw_ent = " ".join(current_entity)
                clean_ent = re.sub(r'\s+([,.:;?!\-\(\)/])', r'\1', raw_ent)
                clean_ent = re.sub(r'([,.:;?!\-\(\)/])\s+', r'\1', clean_ent).strip()
                entities.append({"text": clean_ent, "label": current_label})

            return entities, conf
        except Exception as e:
            logger.warning("Tensor inference failed: %s", e)
            return [], 0.5

    def extract(self, text: str, paper_id: str = "") -> list[dict[str, Any]]:
        """Extract structured clinical claims from biomedical text."""
        if not self._is_loaded and not self._load_attempted:
            self.load()

        claims = []
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text or "") if s.strip()]

        for index, sentence in enumerate(sentences):
            # 1. Run BioBERT transformer inference
            entities, model_conf = self._infer_entities(sentence)

            # 2. Extract clinical relations & predicates
            match = re.search(
                r"\b(reduce[sd]?|increase[sd]? risk|improve[sd]?|decrease[sd]?|inhibit[sd]?|enhance[sd]?|no (?:effect|difference)|associated with|prolong[sd]?|fails to reduce)\b",
                sentence,
                re.I
            )
            if not match:
                continue

            predicate = match.group(0).lower().replace("_", " ")
            before = sentence[:match.start()].strip()
            after = sentence[match.end():].strip(" .;:")
            after_short = re.split(r';|\bhowever\b|\bnevertheless\b|\bgiven the\b', after, flags=re.I)[0].strip(' .,;:')

            # Determine clinical entities with order-preserving deduplication
            def _dedup_list(lst: list[str]) -> list[str]:
                seen = set()
                out = []
                for x in lst:
                    clean_x = re.sub(r'\s+', ' ', x).strip()
                    low = clean_x.lower()
                    if low not in seen and len(low) > 1:
                        seen.add(low)
                        out.append(clean_x)
                return out

            chem_entities = _dedup_list([e["text"] for e in entities if any(k in e.get("label", "").lower() for k in ["chem", "drug", "med"])])
            dis_entities = _dedup_list([e["text"] for e in entities if any(k in e.get("label", "").lower() for k in ["dis", "out", "eff"])])

            clean_before = re.sub(r"^(?:with|nevertheless|however|furthermore|moreover|consequently|therefore|in addition|overall|specifically|herein|we show that|we found that|results demonstrate that|results show that|it is shown that|together,?\s*these findings indicate that|exploratory subgroup analyses suggested(?: potential)?)\s*,?\s*", "", before, flags=re.I).strip()
            clean_before = re.sub(r"\b(?:projected to|likely to|expected to|shown to|demonstrated to)\b", "", clean_before, flags=re.I).strip()
            clean_before = re.sub(r"\s+(?:vs\.?|versus|and|or|with|at|in|by|to|for|of|from)$", "", clean_before, flags=re.I).strip()

            clean_after = re.sub(r"\s+(?:vs\.?|versus|and|or|with|at|in|by|to|for|of|from)$", "", after_short, flags=re.I).strip()
            if clean_after.count("(") > clean_after.count(")"):
                clean_after += ")"

            subject = ", ".join(chem_entities) if chem_entities else (clean_before or before)[-100:]
            obj = ", ".join(dis_entities) if dis_entities else clean_after[:180]

            if subject and obj:
                claims.append({
                    "id": f"{paper_id}:claim:{index}",
                    "paper_id": paper_id,
                    "subject": subject,
                    "predicate": predicate,
                    "object": obj,
                    "extraction_confidence": round(model_conf if entities else 0.85, 3),
                    "model": self.model_name if self._is_loaded else "BioBERT-RuleEngine-Hybrid",
                    "entities": entities,
                    "pico": {
                        "intervention": subject,
                        "outcome": obj
                    }
                })

        return claims
