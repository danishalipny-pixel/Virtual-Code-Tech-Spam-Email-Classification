"""Unit tests for dataset integrity, trained models, and performance benchmarks."""

import json
from pathlib import Path
import joblib
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline


def test_processed_dataset_integrity():
    """Verify that the cleaned dataset exists, has valid columns, and contains records."""
    csv_path = Path("data/processed/emails_cleaned.csv")
    assert csv_path.exists(), f"Processed dataset missing at {csv_path}"

    df = pd.read_csv(csv_path)
    assert len(df) > 1000, "Dataset contains fewer records than expected"
    required_cols = {"text", "label", "label_name"}
    assert required_cols.issubset(df.columns), f"Missing required columns in dataset: {required_cols}"

    # Verify no nulls in text or label
    assert df["text"].isnull().sum() == 0, "Null values found in email text"
    assert df["label"].isnull().sum() == 0, "Null values found in email label"

    # Verify binary classes (0 and 1)
    unique_labels = set(df["label"].unique())
    assert unique_labels == {0, 1}, f"Unexpected labels in dataset: {unique_labels}"

    # Verify deduplication
    assert df["text"].duplicated().sum() == 0, "Found duplicate email texts in processed dataset"


def test_champion_model_file_exists():
    """Verify that the champion model exists on disk and is a Scikit-Learn Pipeline."""
    model_path = Path("models/spam_classifier.joblib")
    assert model_path.exists(), f"Champion model missing at {model_path}"

    model = joblib.load(model_path)
    assert isinstance(model, Pipeline), "Model is not a Scikit-Learn Pipeline"
    assert "tfidf" in model.named_steps, "Pipeline missing 'tfidf' step"
    assert "classifier" in model.named_steps, "Pipeline missing 'classifier' step"


def test_evaluation_json_metrics():
    """Verify that evaluation metadata file exists with valid, non-trivial metrics."""
    json_path = Path("reports/model_evaluation.json")
    assert json_path.exists(), f"Evaluation report missing at {json_path}"

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "best_model" in data
    assert "models" in data
    best_name = data["best_model"]
    metrics = data["models"][best_name]

    # Verify high-performance metrics on genuine SpamAssassin corpus (>95% across the board)
    assert metrics["accuracy"] >= 0.95, f"Accuracy too low: {metrics['accuracy']}"
    assert metrics["precision"] >= 0.95, f"Precision too low: {metrics['precision']}"
    assert metrics["recall"] >= 0.95, f"Recall too low: {metrics['recall']}"
    assert metrics["f1_score"] >= 0.95, f"F1-score too low: {metrics['f1_score']}"
    assert metrics["roc_auc"] >= 0.98, f"ROC-AUC too low: {metrics['roc_auc']}"


def test_evaluation_figures_exist():
    """Verify that all required evaluation charts were generated and saved."""
    figures_dir = Path("reports/figures")
    required_figures = [
        "dataset_distribution.png",
        "confusion_matrix.png",
        "model_comparison.png",
        "metrics_comparison.png",
    ]
    for fig_name in required_figures:
        fig_path = figures_dir / fig_name
        assert fig_path.exists(), f"Missing figure: {fig_path}"
        assert fig_path.stat().st_size > 5000, f"Figure file {fig_name} appears empty or corrupted"
