import re
import json
import logging
from difflib import SequenceMatcher
from typing import Any

from groq import Groq
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from utils.model_config import ordered_models, DEFAULT_CLAIM_EXTRACTION_MODEL
from utils.api_keys import ApiKeyManager

logger = logging.getLogger(__name__)

# Lexical thresholds
GROUNDED_THRESHOLD = 0.60   # >=60% terms found -> grounded
AMBIGUOUS_THRESHOLD = 0.40  # 40-60% -> ambiguous -> LLM check
# <40% -> hallucinated -> drop

GROUNDING_VERIFY_PROMPT = """
You are a scientific fact-checking agent.

A claim was extracted from a source text. Determine if the claim is grounded in the source text.

A claim is GROUNDED if:
- Its core assertion is explicitly stated or directly implied by the source text
- Key entities, numbers, and relationships in the claim appear in the source

A claim is HALLUCINATED if:
- It introduces facts, numbers, or relationships not present in the source
- It overstates or contradicts what the source actually says

Source Text:
{source_text}

Extracted Claim:
{claim}

Return ONLY valid JSON:
{{
  "grounded": true or false,
  "reasoning": "<one sentence explanation>"
}}
"""

EVIDENCE_VERIFY_PROMPT = """
You are a scientific fact-checking agent.

A reasoning statement was generated to justify classifying a paper's abstract as supporting, contradicting, or being inconclusive with respect to a claim.

Determine if the reasoning is grounded in the abstract.

Reasoning is GROUNDED if the abstract actually contains the information cited.
Reasoning is HALLUCINATED if it references facts not present in the abstract.

Abstract:
{abstract}

Reasoning:
{reasoning}

Return ONLY valid JSON:
{{
  "grounded": true or false,
  "reasoning": "<one sentence explanation>"
}}
"""


def _extract_key_terms(text: str) -> list[str]:
    # lowercase, remove punctuation, extract meaningful words
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    words = text.split()
    stopwords = set(ENGLISH_STOP_WORDS)
    return [w for w in words if w not in stopwords and len(w) > 2]


def _split_into_chunks(text: str, chunk_sentences: int = 3) -> list[str]:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if not sentences:
        return [text.strip()] if text.strip() else []

    chunks: list[str] = []
    for i in range(0, len(sentences), chunk_sentences):
        chunk = " ".join(sentences[i : i + chunk_sentences]).strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def _lexical_overlap(claim: str, source_text: str) -> float:
    claim_terms = set(_extract_key_terms(claim))
    source_terms = set(_extract_key_terms(source_text))
    if not claim_terms:
        return 1.0  # empty claim, skip check

    # Exact + fuzzy term matching for simple synonym/morphology robustness.
    matched_score = 0.0
    for claim_term in claim_terms:
        if claim_term in source_terms:
            matched_score += 1.0
            continue
        if any(SequenceMatcher(None, claim_term, src_term).ratio() >= 0.86 for src_term in source_terms):
            matched_score += 0.75

    term_score = matched_score / len(claim_terms)

    claim_compact = " ".join(sorted(claim_terms))
    source_compact = " ".join(sorted(source_terms))
    phrase_score = SequenceMatcher(None, claim_compact, source_compact).ratio()

    return max(term_score, 0.7 * term_score + 0.3 * phrase_score)


def _max_local_overlap(claim: str, source_text: str) -> float:
    chunks = _split_into_chunks(source_text, chunk_sentences=3)
    if not chunks:
        return 0.0
    return max(_lexical_overlap(claim, chunk) for chunk in chunks)


def _confidence_tier(score: float, method: str, grounded: bool) -> str:
    if not grounded:
        return "low"
    if method == "lexical":
        if score >= 0.75:
            return "high"
        if score >= 0.55:
            return "medium"
        return "low"
    if method == "llm":
        return "high" if score >= 0.65 else "medium"
    return "low"


def _normalize_for_match(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


class HallucinationGuard:
    def __init__(
        self,
        api_key: str | list[str],
        model: str = DEFAULT_CLAIM_EXTRACTION_MODEL,
        debug: bool = True,
    ):
        self.key_manager = ApiKeyManager.from_value(api_key)
        self.models = ordered_models(model)
        self.client = Groq(api_key=self.key_manager.current)
        self.debug = debug

    def _call_llm(self, prompt: str) -> dict[str, Any] | None:
        last_error = None
        for model in self.models:
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,  # deterministic for verification
                )
                raw = response.choices[0].message.content.strip()
                break
            except Exception as e:
                last_error = str(e)
                if "rate_limit" in str(e).lower() or "429" in str(e):
                    if self.key_manager.rotate():
                        self.client = Groq(api_key=self.key_manager.current)
                continue
        else:
            logger.warning(f"[HallucinationGuard] All models failed: {last_error}")
            return None

        # parse
        if "```" in raw:
            parts = raw.split("```")
            if len(parts) >= 2:
                raw = parts[1].strip()
            if raw.startswith("json"):
                raw = raw[4:].strip()
        raw = raw.strip()

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.warning(f"[HallucinationGuard] JSON parse failed: {raw}")
            return None

    def verify_claim(self, claim: str, source_text: str) -> dict[str, Any]:
        """
        V1: Verify claim is grounded in source text.
        Returns: {"grounded": bool, "score": float, "method": "lexical|llm", "reasoning": str, "confidence": str}
        """
        if not claim or not source_text:
            return {
                "grounded": False,
                "score": 0.0,
                "method": "lexical",
                "reasoning": "Empty claim or source",
                "confidence": "low",
            }

        score = _max_local_overlap(claim, source_text)

        if self.debug:
            logger.debug("[HallucinationGuard V1] Max local overlap: %.2f", score)

        # clearly grounded
        if score >= GROUNDED_THRESHOLD:
            return {
                "grounded": True,
                "score": score,
                "method": "lexical",
                "reasoning": "Sufficient lexical overlap",
                "confidence": _confidence_tier(score, "lexical", True),
            }

        # clearly hallucinated
        if score < AMBIGUOUS_THRESHOLD:
            if self.debug:
                logger.debug("[HallucinationGuard V1] Low overlap (%.2f) -> hallucinated", score)
            return {
                "grounded": False,
                "score": score,
                "method": "lexical",
                "reasoning": "Insufficient lexical overlap",
                "confidence": "low",
            }

        # ambiguous -> LLM
        if self.debug:
            logger.debug("[HallucinationGuard V1] Ambiguous (%.2f) -> LLM verification", score)

        prompt = GROUNDING_VERIFY_PROMPT.format(
            source_text=source_text[:3000],  # cap to avoid token overflow
            claim=claim,
        )
        result = self._call_llm(prompt)

        if result is None:
            return {
                "grounded": False,
                "score": score,
                "method": "llm_failed",
                "reasoning": "LLM verification failed, marking as unverified",
                "confidence": "low",
            }

        grounded = bool(result.get("grounded", False))
        return {
            "grounded": grounded,
            "score": score,
            "method": "llm",
            "reasoning": result.get("reasoning", ""),
            "confidence": _confidence_tier(score, "llm", grounded),
        }

    def verify_evidence_reasoning(self, reasoning: str, abstract: str) -> dict[str, Any]:
        """
        V2: Verify evidence reasoning is grounded in the paper abstract.
        Returns: {"grounded": bool, "score": float, "method": "lexical|llm", "reasoning": str, "confidence": str}
        """
        if not reasoning or not abstract:
            return {
                "grounded": False,
                "score": 0.0,
                "method": "lexical",
                "reasoning": "Empty reasoning or abstract",
                "confidence": "low",
            }

        score = _max_local_overlap(reasoning, abstract)

        if self.debug:
            logger.debug("[HallucinationGuard V2] Max local overlap: %.2f", score)

        if score >= GROUNDED_THRESHOLD:
            return {
                "grounded": True,
                "score": score,
                "method": "lexical",
                "reasoning": "Sufficient lexical overlap",
                "confidence": _confidence_tier(score, "lexical", True),
            }

        if score < AMBIGUOUS_THRESHOLD:
            return {
                "grounded": False,
                "score": score,
                "method": "lexical",
                "reasoning": "Insufficient lexical overlap",
                "confidence": "low",
            }

        # ambiguous -> LLM
        if self.debug:
            logger.debug("[HallucinationGuard V2] Ambiguous (%.2f) -> LLM verification", score)

        prompt = EVIDENCE_VERIFY_PROMPT.format(
            abstract=abstract[:2000],
            reasoning=reasoning,
        )
        result = self._call_llm(prompt)

        if result is None:
            return {
                "grounded": False,
                "score": score,
                "method": "llm_failed",
                "reasoning": "LLM verification failed, marking as unverified",
                "confidence": "low",
            }

        grounded = bool(result.get("grounded", False))
        return {
            "grounded": grounded,
            "score": score,
            "method": "llm",
            "reasoning": result.get("reasoning", ""),
            "confidence": _confidence_tier(score, "llm", grounded),
        }

    def verify_evidence_span_reasoning(
        self,
        reasoning: str,
        abstract: str,
        evidence_span: str,
    ) -> dict[str, Any]:
        """
        Verify that both reasoning is grounded and evidence_span is present in abstract.
        """
        span = (evidence_span or "").strip()
        if not span:
            return {
                "grounded": False,
                "score": 0.0,
                "method": "span_missing",
                "reasoning": "Missing evidence span",
                "confidence": "low",
            }

        abstract_norm = _normalize_for_match(abstract)
        span_norm = _normalize_for_match(span)
        if span_norm not in abstract_norm:
            return {
                "grounded": False,
                "score": 0.0,
                "method": "span_not_found",
                "reasoning": "Evidence span not found in abstract",
                "confidence": "low",
            }

        base = self.verify_evidence_reasoning(reasoning=reasoning, abstract=abstract)
        if not base.get("grounded", False):
            return base

        return {
            "grounded": True,
            "score": max(base.get("score", 0.0), 0.75),
            "method": f"{base.get('method', 'lexical')}+span",
            "reasoning": base.get("reasoning", "Grounded reasoning with verified span"),
            "confidence": "high" if base.get("confidence") == "high" else "medium",
        }
