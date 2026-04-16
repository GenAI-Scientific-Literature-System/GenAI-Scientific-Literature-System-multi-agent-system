"""Runtime configuration for MERLIN API and agents."""

import os


API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "5000"))
DEBUG = os.getenv("DEBUG", "false").lower() == "true"


def get_all_groq_api_keys():
    keys = []
    for i in range(1, 10):
        key = os.getenv(f"GROQ_API_KEY_{i}")
        if key:
            keys.append(key)
    
    key = os.getenv("GROQ_API_KEY")
    if key:
        keys.append(key)
        
    return list(set(keys))

GROQ_API_KEYS = get_all_groq_api_keys()
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


def get_groq_fallback_models(primary_model: str) -> list[str]:
    """Return ordered fallback models including the primary model first."""
    raw = os.getenv(
        "GROQ_FALLBACK_MODELS",
        "openai/gpt-oss-120b,llama-3.1-8b-instant,mixtral-8x7b-32768",
    )
    fallbacks = [m.strip() for m in raw.split(",") if m.strip()]

    ordered = [primary_model] + fallbacks
    seen = set()
    deduped = []
    for model in ordered:
        if model not in seen:
            seen.add(model)
            deduped.append(model)
    return deduped


GROQ_FALLBACK_MODELS = get_groq_fallback_models(GROQ_MODEL)
GROQ_MAX_TOKENS = int(os.getenv("GROQ_MAX_TOKENS", "8000"))
GROQ_TEMP = float(os.getenv("GROQ_TEMP", "0.1"))


CLAIM_PROMPT = (
    "Extract scientific claims from the following text. "
    "Return JSON only with key 'claims' as an array of objects: "
    "subject, predicate, object, method, domain. "
    "Preserve claim polarity exactly (e.g., keep negations like 'no', 'not', "
    "'fails to', 'insufficient evidence', and support/refute language).\n\n"
    "Text:\n{text}"
)

EVIDENCE_PROMPT = (
    "Given the claims and text, attribute evidence spans per claim. "
    "Return JSON only with key 'evidence' as an array of objects: "
    "claim_id (int), spans (array of strings), strength (high|medium|low).\n\n"
    "Claims:\n{claims}\n\nText:\n{text}"
)

ASSUMPTION_PROMPT = (
    "Extract assumptions from the text. "
    "Return JSON only with key 'assumptions' as an array of objects: "
    "type, constraint, explicit, span.\n\n"
    "Text:\n{text}"
)


UNCERTAINTY_CONFLICT_WEIGHT = float(os.getenv("UNCERTAINTY_CONFLICT_WEIGHT", "0.4"))
UNCERTAINTY_EVIDENCE_WEIGHT = float(os.getenv("UNCERTAINTY_EVIDENCE_WEIGHT", "0.3"))
UNCERTAINTY_STABILITY_WEIGHT = float(os.getenv("UNCERTAINTY_STABILITY_WEIGHT", "0.3"))


NLI_THRESHOLD = float(os.getenv("NLI_THRESHOLD", "0.65"))
NLI_MODEL_PRIMARY = os.getenv("NLI_MODEL_PRIMARY", "facebook/bart-large-mnli")
