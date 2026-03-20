DEFAULT_CLAIM_EXTRACTION_MODEL = "llama-3.3-70b-versatile"
DEFAULT_EVIDENCE_COLLECTION_MODEL = "llama-3.3-70b-versatile"

CLAIM_EXTRACTION_FALLBACK_MODELS = [
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-120b",
    "moonshotai/kimi-k2-instruct-0905",
    "moonshotai/kimi-k2-instruct",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "qwen/qwen3-32b",
    "openai/gpt-oss-20b",
    "llama-3.1-8b-instant",
]

EVIDENCE_COLLECTION_FALLBACK_MODELS = [
    "openai/gpt-oss-120b",
    "llama-3.3-70b-versatile",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "moonshotai/kimi-k2-instruct-0905",
    "openai/gpt-oss-20b",
]


def ordered_models(primary_model: str, fallback_models: list[str] | None = None) -> list[str]:
    fallbacks = fallback_models or CLAIM_EXTRACTION_FALLBACK_MODELS
    return [primary_model] + [model for model in fallbacks if model != primary_model]
