"""
MERLIN Document Store
━━━━━━━━━━━━━━━━━━━━
After the first pipeline run for a document, the full MERLIN_STRUCT is cached
in memory. Subsequent runs reuse the parsed structure — raw text never leaves
the extraction step.

MERLIN_STRUCT = {
    "doc_id":      str,          # sha256[:16] of raw text
    "paper_id":    str,          # filename / label
    "claims":      [dict, ...],  # serialised Claim objects
    "assumptions": [dict, ...],  # serialised Assumption objects
    "chunk_count": int,
    "token_saved": int,          # estimated tokens saved vs raw text
}
"""
import hashlib
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

_STORE: Dict[str, Dict[str, Any]] = {}


def doc_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def get_struct(text: str) -> Optional[Dict]:
    did = doc_id(text)
    hit = _STORE.get(did)
    if hit:
        logger.info("DocumentStore: cache HIT %s (%d claims, %d assumptions)",
                    did[:8], len(hit.get("claims", [])), len(hit.get("assumptions", [])))
    return hit


def put_struct(text: str, paper_id: str, claims: list, assumptions: list, chunk_count: int):
    did = doc_id(text)
    saved_tokens = max(0, len(text) // 4 - sum(len(str(c)) // 4 for c in claims))
    _STORE[did] = {
        "doc_id":      did,
        "paper_id":    paper_id,
        "claims":      claims,
        "assumptions": assumptions,
        "chunk_count": chunk_count,
        "token_saved": saved_tokens,
    }
    logger.info("DocumentStore: cached %s — %d claims, ~%d tokens saved",
                did[:8], len(claims), saved_tokens)


def clear():
    global _STORE
    _STORE = {}
    logger.info("DocumentStore: cleared.")
