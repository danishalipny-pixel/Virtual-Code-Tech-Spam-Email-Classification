"""Unit tests for text preprocessing and NLP utilities."""

import pytest
from src.preprocessing import (
    PRESERVED_STOP_WORDS,
    TextPreprocessor,
    clean_email_text,
)


def test_html_tag_removal():
    """Verify that HTML tags and markup are stripped cleanly."""
    raw = "<p>Hello <b>World</b>! <a href='https://example.com'>Click Here</a></p>"
    cleaned = clean_email_text(raw)
    assert "<p>" not in cleaned
    assert "<b>" not in cleaned
    assert "</a>" not in cleaned
    assert "hello" in cleaned
    assert "world" in cleaned


def test_html_entity_decoding():
    """Verify that HTML entities like &amp; &lt; &#39; are properly decoded."""
    raw = "Fish &amp; Chips &lt;dinner&gt; it&#39;s delicious"
    cleaned = clean_email_text(raw)
    assert "&amp;" not in cleaned
    assert "fish" in cleaned
    assert "chips" in cleaned


def test_url_normalization():
    """Verify that HTTP, HTTPS, and WWW URLs are mapped to 'httpaddr' token."""
    raw = "Visit https://secure-bank-login.xyz/auth or www.promo-claim.com today!"
    cleaned = clean_email_text(raw)
    assert "httpaddr" in cleaned
    assert "https://" not in cleaned
    assert "www.promo-claim.com" not in cleaned


def test_email_normalization():
    """Verify that email addresses are mapped to 'emailaddr' token."""
    raw = "Contact support@service-desk.net or billing@corp.org for help."
    cleaned = clean_email_text(raw)
    assert "emailaddr" in cleaned
    assert "support@service-desk.net" not in cleaned


def test_currency_normalization():
    """Verify that currency symbols ($, £, €, ¥) are mapped to 'moneysymb'."""
    raw = "Earn $5000 or £3500 and €4000 bonus cash!"
    cleaned = clean_email_text(raw)
    assert "moneysymb" in cleaned
    assert "$" not in cleaned
    assert "£" not in cleaned
    assert "€" not in cleaned


def test_negation_word_preservation():
    """Verify that essential negation words ('not', 'no', 'never') are retained."""
    raw = "This is NOT a fraudulent offer. Never reply to untrusted senders."
    cleaned = clean_email_text(raw)
    assert "not" in cleaned.split()
    assert "never" in cleaned.split()
    assert "not" not in PRESERVED_STOP_WORDS
    assert "never" not in PRESERVED_STOP_WORDS


def test_empty_and_none_safety():
    """Verify that None, empty strings, and whitespace return clean empty strings."""
    assert clean_email_text("") == ""
    assert clean_email_text("   \n\t  ") == ""
    assert clean_email_text(None) == ""


def test_whitespace_normalization():
    """Verify that excessive tabs, newlines, and spaces collapse into single spaces."""
    raw = "Line  1\n\n\nLine\t\t2    Line 3"
    cleaned = clean_email_text(raw)
    assert "   " not in cleaned
    assert "\n" not in cleaned
    assert "\t" not in cleaned
    assert cleaned == "line 1 line 2 line 3"


def test_text_preprocessor_transformer():
    """Verify Scikit-Learn transformer interface compatibility."""
    transformer = TextPreprocessor()
    transformer.fit(["dummy text"])
    transformed = transformer.transform(["<p>Test $100</p>", "Visit https://site.com"])
    assert isinstance(transformed, list)
    assert len(transformed) == 2
    assert "moneysymb" in transformed[0]
    assert "httpaddr" in transformed[1]
