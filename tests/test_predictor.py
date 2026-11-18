"""Unit tests for the EmailSpamPredictor service and inference pipeline."""

import pytest
from src.predictor import EmailSpamPredictor, classify_email


@pytest.fixture(scope="module")
def predictor():
    """Module-level predictor fixture."""
    return EmailSpamPredictor()


def test_assignment_legitimate_email_example(predictor):
    """Test exact legitimate email example from assignment prompt."""
    text = "Hello, your appointment has been confirmed for tomorrow at 10 AM. Please bring your documents."
    result = predictor.predict(text)

    assert result["is_valid"] is True
    assert result["is_spam"] is False
    assert result["status_category"] == "NOT SPAM"
    assert result["label_display"] == "LEGITIMATE EMAIL"
    assert 0.0 <= result["spam_probability"] <= 0.50
    assert 0.50 <= result["ham_probability"] <= 1.0


def test_assignment_spam_email_example(predictor):
    """Test exact spam email example from assignment prompt."""
    text = "Congratulations! You have won a free prize. Claim your reward immediately by clicking this link."
    result = predictor.predict(text)

    assert result["is_valid"] is True
    assert result["is_spam"] is True
    assert result["status_category"] == "SPAM"
    assert result["label_display"] == "SPAM EMAIL DETECTED"
    assert result["spam_probability"] >= 0.60
    assert len(result["detected_triggers"]) > 0


def test_empty_string_validation(predictor):
    """Verify that empty email string returns validation error."""
    result = predictor.predict("")
    assert result["is_valid"] is False
    assert "error" in result


def test_whitespace_string_validation(predictor):
    """Verify that whitespace-only email string returns validation error."""
    result = predictor.predict("   \n\t   ")
    assert result["is_valid"] is False
    assert "error" in result


def test_none_input_validation(predictor):
    """Verify that None input returns validation error safely."""
    result = predictor.predict(None)
    assert result["is_valid"] is False
    assert "error" in result


def test_business_communication_classification(predictor):
    """Verify standard legitimate business correspondence."""
    text = (
        "Hi team, please find attached the quarterly review report. "
        "We will hold our sync tomorrow at 2 PM in Conference Room B."
    )
    result = predictor.predict(text)
    assert result["is_valid"] is True
    assert result["is_spam"] is False
    assert result["status_category"] == "NOT SPAM"


def test_financial_lottery_scam_classification(predictor):
    """Verify high-risk financial lottery scam is detected as spam."""
    text = (
        "URGENT: You are the lucky winner of $2,500,000 cash fund! "
        "Contact claim-agent@lottery-payout.biz immediately to receive your payment."
    )
    result = predictor.predict(text)
    assert result["is_valid"] is True
    assert result["is_spam"] is True
    assert result["status_category"] == "SPAM"
    assert "Currency symbols / Financial indicators" in result["detected_triggers"]


def test_probability_sums_to_one(predictor):
    """Verify that probabilities sum to 1.0 within floating point tolerance."""
    text = "Meeting rescheduled to next Wednesday afternoon."
    result = predictor.predict(text)
    assert result["is_valid"] is True
    total_prob = result["ham_probability"] + result["spam_probability"]
    assert pytest.approx(total_prob, abs=1e-3) == 1.0


def test_convenience_function():
    """Verify the top-level classify_email convenience function."""
    result = classify_email("Hi Sarah, here are the project notes from our discussion.")
    assert result["is_valid"] is True
    assert result["is_spam"] is False
