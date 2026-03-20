import os
import re
import json
import hashlib
import logging
from io import BytesIO

from typing import Any

import nltk
import numpy as np
import requests
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

CACHE_DIR = ".cache/document_retriever"
CHUNK_SIZE = 5
CHUNK_OVERLAP = 2
MAX_PAGES = 20
SPECTER_MODEL = "allenai-specter"


def _ensure_nltk() -> None:
	try:
		nltk.data.find("tokenizers/punkt_tab")
	except LookupError:
		nltk.download("punkt_tab", quiet=True)


def _cache_path(key: str) -> str:
	os.makedirs(CACHE_DIR, exist_ok=True)
	return os.path.join(CACHE_DIR, f"{key}.json")


def _cache_get(key: str) -> dict[str, Any] | None:
	path = _cache_path(key)
	if os.path.exists(path):
		try:
			with open(path, "r") as f:
				return json.load(f)
		except Exception:
			pass
	return None


def _cache_set(key: str, data: dict[str, Any]) -> None:
	try:
		with open(_cache_path(key), "w") as f:
			json.dump(data, f)
	except Exception as e:
		logger.warning(f"[DocumentRetriever] Cache write failed: {e}")


def _url_to_key(url: str) -> str:
	return hashlib.sha1(url.encode()).hexdigest()[:16]


def fetch_pdf_bytes(url: str, timeout: int = 20) -> bytes | None:
	try:
		response = requests.get(
			url,
			timeout=timeout,
			headers={
				"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
				"Accept": "application/pdf,*/*",
				"Accept-Language": "en-US,en;q=0.9",
				"Referer": "https://europepmc.org/",
			},
		)
		response.raise_for_status()
		return response.content
	except Exception as e:
		logger.warning(f"[DocumentRetriever] PDF fetch failed ({url}): {e}")
		return None


def extract_text_from_bytes(pdf_bytes: bytes, max_pages: int = MAX_PAGES) -> str:
	try:
		reader = PdfReader(BytesIO(pdf_bytes))
		pages = reader.pages[:max_pages]
		text = "\n".join(page.extract_text() or "" for page in pages)
		text = re.sub(r"\s+", " ", text).strip()
		return text
	except Exception as e:
		logger.warning(f"[DocumentRetriever] PDF extraction failed: {e}")
		return ""


def extract_text_from_file(file_bytes: bytes) -> str:
	return extract_text_from_bytes(file_bytes)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
	_ensure_nltk()
	sentences = nltk.sent_tokenize(text)
	if not sentences:
		return []

	chunks: list[str] = []
	i = 0
	while i < len(sentences):
		chunk = sentences[i : i + chunk_size]
		chunks.append(" ".join(chunk))
		step = max(1, chunk_size - overlap)
		i += step

	return [c for c in chunks if len(c.strip()) > 50]


class DocumentRetriever:
	def __init__(
		self,
		model: SentenceTransformer | None = None,
		chunk_size: int = CHUNK_SIZE,
		overlap: int = CHUNK_OVERLAP,
		debug: bool = True,
	):
		self.model = model or SentenceTransformer(SPECTER_MODEL)
		self.chunk_size = chunk_size
		self.overlap = overlap
		self.debug = debug

		self._chunks: list[str] = []
		self._embeddings: np.ndarray | None = None

	def load_from_url(self, url: str) -> bool:
		cache_key = _url_to_key(url)
		cached = _cache_get(cache_key)

		if cached:
			if self.debug:
				print(f"[DocumentRetriever] Cache hit: {url}")
			self._chunks = cached.get("chunks", [])
			self._embeddings = np.array(cached.get("embeddings", []), dtype=np.float32)
			return bool(len(self._chunks) and self._embeddings.size)

		if self.debug:
			print(f"[DocumentRetriever] Fetching PDF: {url}")

		pdf_bytes = fetch_pdf_bytes(url)
		if not pdf_bytes:
			return False

		return self._process_and_cache(pdf_bytes, cache_key)

	def load_from_bytes(self, file_bytes: bytes, file_id: str = "") -> bool:
		cache_key = _url_to_key(file_id or hashlib.sha1(file_bytes[:512]).hexdigest())
		cached = _cache_get(cache_key)

		if cached:
			if self.debug:
				print(f"[DocumentRetriever] Cache hit: {file_id}")
			self._chunks = cached.get("chunks", [])
			self._embeddings = np.array(cached.get("embeddings", []), dtype=np.float32)
			return bool(len(self._chunks) and self._embeddings.size)

		return self._process_and_cache(file_bytes, cache_key)

	def _process_and_cache(self, pdf_bytes: bytes, cache_key: str) -> bool:
		text = extract_text_from_bytes(pdf_bytes)
		if not text:
			return False

		self._chunks = chunk_text(text, self.chunk_size, self.overlap)
		if not self._chunks:
			return False

		if self.debug:
			print(f"[DocumentRetriever] {len(self._chunks)} chunks generated")

		self._embeddings = self.model.encode(
			self._chunks,
			batch_size=16,
			convert_to_numpy=True,
			normalize_embeddings=True,
			show_progress_bar=False,
		)

		_cache_set(
			cache_key,
			{
				"chunks": self._chunks,
				"embeddings": self._embeddings.tolist(),
			},
		)
		return True

	def retrieve(self, query: str, top_k: int = 4) -> str:
		if self._embeddings is None or len(self._chunks) == 0:
			return ""

		query_emb = self.model.encode(
			[query],
			convert_to_numpy=True,
			normalize_embeddings=True,
		)[0]

		scores = np.dot(self._embeddings, query_emb)
		top_indices = np.argsort(scores)[::-1][:top_k]
		top_chunks = [self._chunks[i] for i in sorted(top_indices)]
		return " ".join(top_chunks)

	def is_loaded(self) -> bool:
		return self._embeddings is not None and len(self._chunks) > 0
