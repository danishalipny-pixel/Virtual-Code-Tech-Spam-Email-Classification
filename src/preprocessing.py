"""Text Preprocessing and Natural Language Processing module.

Hidden Virtual CodeTech - Artificial Intelligence / Generative AI Internship
Project: Spam or Not Spam – Email Classification System
"""

import html
import re
from typing import Any, List, Union
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Precompiled regex patterns for performance and consistency
RE_HTML_TAGS = re.compile(r"<[^>]+>", re.IGNORECASE)
RE_URL = re.compile(
    r"(?:https?://|www\.|ftp://)[\w\d\.\-_~:/?#\[\]@!$&'()*+,;=%]+",
    re.IGNORECASE,
)
RE_EMAIL = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
)
RE_CURRENCY = re.compile(r"[\$£€¥₹]")
RE_PUNCTUATION_REPEAT = re.compile(r"([!?]){2,}")
RE_SPECIAL_CHARS = re.compile(r"[^a-zA-Z0-9\s]")
RE_WHITESPACE = re.compile(r"\s+")

# Custom stop words list that preserves vital negation words as required by assignment NLP rules
NEGATION_WORDS = {"not", "no", "never", "cannot", "neither", "nor"}
PRESERVED_STOP_WORDS = list(ENGLISH_STOP_WORDS - NEGATION_WORDS)


def clean_email_text(text: Any) -> str:
    """Preprocess and normalize email text for TF-IDF feature extraction.

    Preserves semantic meaning, maps key spam indicators (links, currency, emails)
    to normalized tokens, and safely strips markup and punctuation.

    Args:
        text: Raw email text or string-convertible object.

    Returns:
        str: Cleaned and normalized text representation.
    """
    if text is None:
        return ""

    if not isinstance(text, str):
        text = str(text)

    # 1. Decode HTML entities (e.g., &amp;, &lt;, &#39;)
    text = html.unescape(text)

    # 2. Strip HTML tags
    text = RE_HTML_TAGS.sub(" ", text)

    # 3. Lowercase for uniform feature matching
    text = text.lower()

    # 4. Standardize URLs -> httpaddr token (links are key spam indicators)
    text = RE_URL.sub(" httpaddr ", text)

    # 5. Standardize email addresses -> emailaddr token
    text = RE_EMAIL.sub(" emailaddr ", text)

    # 6. Standardize currency symbols -> moneysymb token (financial triggers)
    text = RE_CURRENCY.sub(" moneysymb ", text)

    # 7. Normalize repeated punctuation (! and ?)
    text = RE_PUNCTUATION_REPEAT.sub(r" \1 ", text)

    # 8. Clean remaining non-alphanumeric noise characters, preserving whitespace
    text = RE_SPECIAL_CHARS.sub(" ", text)

    # 9. Normalize whitespace (remove multiple spaces, tabs, newlines)
    text = RE_WHITESPACE.sub(" ", text).strip()

    return text


class TextPreprocessor(BaseEstimator, TransformerMixin):
    """Scikit-Learn compatible transformer for pipeline integration."""

    def fit(self, X: Any, y: Any = None):
        return self

    def transform(self, X: Union[List[str], Any]) -> List[str]:
        if hasattr(X, "__iter__") and not isinstance(X, str):
            return [clean_email_text(item) for item in X]
        return [clean_email_text(X)]


if __name__ == "__main__":
    sample = (
        "<html><body><h1>CONGRATULATIONS!!</h1><p>You won $1,000,000 CASH prize! "
        "Claim now at https://prize-claim.example.com or email support@free-money.biz. "
        "This is NOT a drill!!!</p></body></html>"
    )
    print("Original Sample:")
    print(sample)
    print("\nCleaned Sample:")
    print(clean_email_text(sample))
