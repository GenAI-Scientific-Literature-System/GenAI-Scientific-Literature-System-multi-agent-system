"""
MERLIN Retrieval Layer (RAG)
━━━━━━━━━━━━━━━━━━━━━━━━━━━
Splits each document into overlapping chunks, builds a TF-IDF index,
then lets each agent retrieve only the relevant fragment instead of
sending the full document text.

Token reduction: 60–90 % vs sending full sanitised text.
Zero new dependencies — uses sklearn (already in requirements.txt).
"""
import re
import logging
from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

CHUNK_CHARS   = 350   # target characters per chunk
CHUNK_OVERLAP = 80    # overlap so claims don't get split mid-sentence
DEFAULT_TOP_K = 3


# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_text(text: str, size: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Split at sentence boundaries into overlapping character-level chunks.
    Sentence-boundary split → cleaner context windows for the LLM.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks, current, cur_len = [], [], 0

    for sent in sentences:
        if cur_len + len(sent) > size and current:
            chunks.append(" ".join(current))
            # slide window: remove from front until under overlap budget
            while current and cur_len > overlap:
                cur_len -= len(current[0]) + 1
                current.pop(0)
        current.append(sent)
        cur_len += len(sent) + 1

    if current:
        chunks.append(" ".join(current))

    return chunks if chunks else [text]


# ── Retriever ─────────────────────────────────────────────────────────────────

class DocumentRetriever:
    """
    Per-document TF-IDF retriever.
    Build once during pipeline init; reuse across all agents for that paper.
    """

    def __init__(self, text: str, paper_id: str = ""):
        self.paper_id = paper_id
        self.chunks   = chunk_text(text)
        self._matrix  = None
        self._vec     = None

        try:
            self._vec    = TfidfVectorizer(stop_words="english", max_features=8000)
            self._matrix = self._vec.fit_transform(self.chunks)
            logger.info("Retriever[%s]: %d chunks indexed.", paper_id[:20], len(self.chunks))
        except Exception as exc:
            logger.warning("Retriever[%s]: TF-IDF build failed (%s). Fallback to head.", paper_id, exc)

    # ── Public API ────────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = DEFAULT_TOP_K) -> str:
        """
        Return the top-k most relevant chunks concatenated.
        Falls back to the first top_k chunks if the index is unavailable.
        """
        if self._matrix is None or not query.strip():
            return " ".join(self.chunks[:top_k])

        try:
            q_vec = self._vec.transform([query])
            sims  = cosine_similarity(q_vec, self._matrix)[0]
            order = np.argsort(sims)[::-1][:top_k]
            # Sort back to document order for coherence
            order = sorted(order, key=lambda i: i)
            return " ".join(self.chunks[i] for i in order)
        except Exception as exc:
            logger.debug("Retriever.retrieve error: %s", exc)
            return " ".join(self.chunks[:top_k])

    def retrieve_for_claim(self, claim_text: str) -> str:
        """Retrieve context most relevant to a single claim."""
        return self.retrieve(claim_text, top_k=2)

    def full_text(self) -> str:
        return " ".join(self.chunks)

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)
