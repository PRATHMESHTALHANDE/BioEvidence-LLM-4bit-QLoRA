"""Text cleaning, unicode normalization, and whitespace stripping for biomedical text."""

import re
import unicodedata


def clean_text(text: str) -> str:
    """Normalize unicode, collapse redundant whitespace, and sanitize biomedical text."""
    if not text:
        return ""

    # NFKC unicode normalization
    text = unicodedata.normalize("NFKC", text)

    # Standardize quotation marks and dashes
    text = re.sub(r'[\u2018\u2019]', "'", text)
    text = re.sub(r'[\u201C\u201D]', '"', text)
    text = re.sub(r'[\u2013\u2014]', '-', text)

    # Replace non-breaking spaces
    text = text.replace('\u00a0', ' ')

    # Collapse multiple spaces and excessive newlines
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()
