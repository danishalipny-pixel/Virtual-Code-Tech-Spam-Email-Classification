"""Model evaluation and report generation module.

Hidden Virtual CodeTech - Artificial Intelligence / Generative AI Internship
Project: Spam or Not Spam – Email Classification System
"""

import json
import logging
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from src.data_loader import load_and_process_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def evaluate_saved_model(
    model_path: Path = Path("models/spam_classifier.joblib"),
    eval_json_path: Path = Path("reports/model_evaluation.json"),
) -> dict:
    """Load the trained model and output detailed performance metrics."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at {model_path}. "
            f"Please run 'python -m src.train_model' first to train the classifier."
        )

    logger.info(f"Loading champion model from {model_path}...")
    pipeline = joblib.load(model_path)

    if eval_json_path.exists():
        with open(eval_json_path, "r", encoding="utf-8") as f:
            eval_data = json.load(f)
        logger.info(f"Loaded existing evaluation metadata from {eval_json_path}")
        return eval_data

    # If JSON doesn't exist, compute evaluation metrics from data
    df = load_and_process_dataset()
    X = df["text"].fillna("").astype(str).tolist()
    y = np.array(df["label"].tolist(), dtype=int)

    y_pred = pipeline.predict(X)
    cm = confusion_matrix(y, y_pred).tolist()
    report = classification_report(y, y_pred, target_names=["Ham", "Spam"], output_dict=True)

    summary = {
        "confusion_matrix": cm,
        "classification_report": report,
    }
    return summary


def print_formatted_summary(eval_data: dict):
    """Print an executive, formatted summary of the model evaluation results."""
    best_model = eval_data.get("best_model", "Selected Model")
    print("=" * 65)
    print(f" SPAMGUARD AI - MODEL PERFORMANCE EVALUATION REPORT")
    print(f" Champion Algorithm: {best_model}")
    print("=" * 65)
    print(f" Total Dataset Samples : {eval_data.get('dataset_size', 'N/A'):,}")
    print(f" Training Samples (80%): {eval_data.get('train_size', 'N/A'):,}")
    print(f" Testing Samples (20%) : {eval_data.get('test_size', 'N/A'):,}")
    print("-" * 65)

    models = eval_data.get("models", {})
    for name, metrics in models.items():
        is_best = " [BEST MODEL]" if name == best_model else ""
        print(f"\nModel: {name}{is_best}")
        print(f"  Accuracy  : {metrics['accuracy']:.4f} ({metrics['accuracy'] * 100:.2f}%)")
        print(f"  Precision : {metrics['precision']:.4f} ({metrics['precision'] * 100:.2f}%)")
        print(f"  Recall    : {metrics['recall']:.4f} ({metrics['recall'] * 100:.2f}%)")
        print(f"  F1-Score  : {metrics['f1_score']:.4f} ({metrics['f1_score'] * 100:.2f}%)")
        print(f"  ROC-AUC   : {metrics.get('roc_auc', 0):.4f}")
        print(f"  Confusion Matrix : {metrics['confusion_matrix']}")

    print("=" * 65)


if __name__ == "__main__":
    eval_data = evaluate_saved_model()
    print_formatted_summary(eval_data)
