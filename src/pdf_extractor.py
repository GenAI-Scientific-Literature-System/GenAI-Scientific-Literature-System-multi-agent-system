"""
MERLIN PDF Extractor
Extracts clean text from uploaded PDF files using PyMuPDF (fitz).
Handles: multi-page, columns, headers/footers, encoding issues.
"""
import re
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Max chars sent to pipeline — keeps tokens sane (~6000 ≈ 1500 tokens)
MAX_CHARS = 8000

try:
    import fitz  # PyMuPDF
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False
    logger.warning("PyMuPDF not installed. PDF upload will be unavailable.")


def _clean(text: str) -> str:
    """
    Post-process extracted text:
    - Collapse excessive whitespace / blank lines
    - Remove page-header noise (page numbers, running titles)
    - Fix common ligature issues (ﬁ → fi, ﬀ → ff)
    """
    # Ligatures
    ligatures = {"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl",
                 "\u2013": "-", "\u2014": "-", "\u2019": "'", "\u201c": '"', "\u201d": '"'}
    for bad, good in ligatures.items():
        text = text.replace(bad, good)

    # Remove lone page numbers (lines that are just a number)
    text = re.sub(r"(?m)^\s*\d{1,4}\s*$", "", text)

    # Collapse 3+ consecutive newlines → 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Collapse runs of spaces
    text = re.sub(r"[ \t]{2,}", " ", text)

    return text.strip()


def extract_text_from_pdf(file_bytes: bytes, filename: str = "upload.pdf") -> dict:
    """
    Extract text from a PDF byte stream.
    Returns:
      {
        "text":      str,          # cleaned full text (capped at MAX_CHARS)
        "pages":     int,          # total pages
        "truncated": bool,         # True if text was capped
        "error":     str | None    # error message if extraction failed
      }
    """
    if not HAS_FITZ:
        return {
            "text": "", "pages": 0, "truncated": False,
            "error": "PyMuPDF not installed. Run: pip install pymupdf"
        }

    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        total_pages = len(doc)

        pages_text = []
        for page_num, page in enumerate(doc):
            # Use "text" mode: preserves reading order, handles columns
            raw = page.get_text("text", sort=True)
            if raw.strip():
                pages_text.append(raw)

        doc.close()

        full_text = "\n\n".join(pages_text)
        full_text = _clean(full_text)

        truncated = len(full_text) > MAX_CHARS
        if truncated:
            # Cut at last sentence boundary before the limit
            cutoff = full_text[:MAX_CHARS].rfind(". ")
            full_text = full_text[: cutoff + 1 if cutoff > 0 else MAX_CHARS]
            logger.info("PDF '%s': text truncated to %d chars.", filename, len(full_text))

        logger.info(
            "PDF '%s': %d pages, %d chars extracted%s.",
            filename, total_pages, len(full_text),
            " (truncated)" if truncated else ""
        )
        return {
            "text": full_text,
            "pages": total_pages,
            "truncated": truncated,
            "error": None,
        }

    except Exception as e:
        logger.error("PDF extraction failed for '%s': %s", filename, e)
        return {
            "text": "", "pages": 0, "truncated": False,
            "error": f"Could not extract text: {str(e)}"
        }


def is_pdf_available() -> bool:
    return HAS_FITZ
