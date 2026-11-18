"""Email Spam Predictor service module.

Hidden Virtual CodeTech - Artificial Intelligence / Generative AI Internship
Project: Spam or Not Spam – Email Classification System
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import joblib
import numpy as np

from src.preprocessing import clean_email_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = Path("models/spam_classifier.joblib")


class EmailSpamPredictor:
    """Predictor class that loads the trained TF-IDF + ML Pipeline

    and performs classification and risk assessment on unseen email text.
    """

    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self.pipeline = None
        self._load_model()

    def _load_model(self):
        """Load the serialized Scikit-learn Pipeline from disk."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Trained model not found at '{self.model_path}'. "
                "Please run 'python -m src.train_model' to train and serialize the model first."
            )
        try:
            self.pipeline = joblib.load(self.model_path)
            logger.info(f"Loaded classifier pipeline from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load model from {self.model_path}: {e}")
            raise RuntimeError(f"Error loading model from {self.model_path}: {e}")

    def predict(self, raw_email_text: str) -> Dict[str, Any]:
        """Perform email classification with confidence scoring and diagnostic features.

        Args:
            raw_email_text: Raw incoming email text to classify.

        Returns:
            Dict[str, Any]: Detailed classification results dictionary.
        """
        # 1. Validation
        if raw_email_text is None:
            return {
                "is_valid": False,
                "error": "Email content is None. Please provide valid text.",
            }

        cleaned_text = clean_email_text(raw_email_text)
        if not cleaned_text.strip():
            return {
                "is_valid": False,
                "error": "Email content is empty or contains only whitespace/non-informative symbols.",
            }

        if self.pipeline is None:
            self._load_model()

        # 2. Pipeline Prediction (Vectorization + Classification)
        prediction_class = int(self.pipeline.predict([raw_email_text])[0])
        is_spam = bool(prediction_class == 1)

        # 3. Probability / Confidence calculation
        probabilities = self.pipeline.predict_proba([raw_email_text])[0]
        ham_prob = float(probabilities[0])
        spam_prob = float(probabilities[1])

        # Confidence is the winning class's probability
        confidence = float(max(ham_prob, spam_prob))

        # 4. Feature and trigger detection
        detected_triggers = []
        if "httpaddr" in cleaned_text:
            detected_triggers.append("Hyperlinks / Web Addresses detected")
        if "moneysymb" in cleaned_text:
            detected_triggers.append("Currency symbols / Financial indicators")
        if "emailaddr" in cleaned_text:
            detected_triggers.append("Email addresses referenced in body")
        if any(term in cleaned_text.split() for term in ["free", "prize", "winner", "congratulations", "urgent", "claim", "credit", "cash"]):
            detected_triggers.append("Promotional / High-urgency keywords detected")

        label_display = "SPAM EMAIL DETECTED" if is_spam else "LEGITIMATE EMAIL"
        status_category = "SPAM" if is_spam else "NOT SPAM"

        return {
            "is_valid": True,
            "prediction": prediction_class,
            "is_spam": is_spam,
            "status_category": status_category,
            "label_display": label_display,
            "confidence": confidence,
            "confidence_percentage": round(confidence * 100, 2),
            "spam_probability": round(spam_prob, 4),
            "ham_probability": round(ham_prob, 4),
            "cleaned_preview": cleaned_text[:300] + ("..." if len(cleaned_text) > 300 else ""),
            "detected_triggers": detected_triggers,
            "disclaimer": (
                "Note: Machine learning predictions are probabilistic inferences based on training patterns "
                "from the Apache SpamAssassin corpus and do not constitute absolute guarantees."
            ),
        }


# Global cached predictor instance
_PREDICTOR_INSTANCE: Optional[EmailSpamPredictor] = None


def get_predictor(model_path: Optional[Path] = None) -> EmailSpamPredictor:
    """Retrieve or initialize the singleton EmailSpamPredictor."""
    global _PREDICTOR_INSTANCE
    if _PREDICTOR_INSTANCE is None:
        _PREDICTOR_INSTANCE = EmailSpamPredictor(model_path)
    return _PREDICTOR_INSTANCE


def classify_email(email_text: str) -> Dict[str, Any]:
    """Convenience helper to classify email using cached predictor."""
    predictor = get_predictor()
    return predictor.predict(email_text)


if __name__ == "__main__":
    predictor = EmailSpamPredictor()

    test_legit = "Hello, your appointment has been confirmed for tomorrow at 10 AM. Please bring your documents."
    test_spam = "Congratulations! You have won a free prize. Claim your reward immediately by clicking this link: http://win-now.com."

    print("--- Legitimate Test Case ---")
    print(predictor.predict(test_legit))

    print("\n--- Spam Test Case ---")
    print(predictor.predict(test_spam))
