DEFAULT_CLAIM_EXTRACTION_MODEL = "llama-3.3-70b-versatile"

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


def ordered_models(primary_model: str, fallback_models: list[str] | None = None) -> list[str]:
    fallbacks = fallback_models or CLAIM_EXTRACTION_FALLBACK_MODELS
    return [primary_model] + [model for model in fallbacks if model != primary_model]
