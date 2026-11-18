"""Model training, tuning, comparison, and serialization module.

Hidden Virtual CodeTech - Artificial Intelligence / Generative AI Internship
Project: Spam or Not Spam – Email Classification System
"""

import json
import logging
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from src.data_loader import load_and_process_dataset
from src.preprocessing import PRESERVED_STOP_WORDS, clean_email_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def generate_evaluation_figures(
    df: pd.DataFrame,
    y_test: np.ndarray,
    models_results: dict,
    best_model_name: str,
    output_dir: Path,
):
    """Generate and save professional evaluation figures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    palette = ["#1E3A8A", "#0D9488"]  # Dark navy & Teal

    # 1. Dataset Class Distribution Chart
    plt.figure(figsize=(7, 5))
    class_counts = df["label_name"].value_counts()
    bars = plt.bar(
        ["Legitimate (Ham)", "Spam"],
        [class_counts.get("ham", 0), class_counts.get("spam", 0)],
        color=palette,
        width=0.55,
        edgecolor="#0F172A",
        linewidth=1.2,
    )
    plt.title("Email Dataset Class Distribution (SpamAssassin Corpus)", fontsize=13, fontweight="bold", pad=15)
    plt.ylabel("Number of Emails", fontsize=11)
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        height = bar.get_height()
        pct = (height / len(df)) * 100
        plt.annotate(
            f"{height:,} ({pct:.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="semibold",
        )
    plt.tight_layout()
    plt.savefig(output_dir / "dataset_distribution.png", dpi=300)
    plt.close()
    logger.info("Saved reports/figures/dataset_distribution.png")

    # 2. Confusion Matrices (Side-by-side)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for i, (name, res) in enumerate(models_results.items()):
        cm = np.array(res["confusion_matrix"])
        ax = axes[i]
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues" if i == 0 else "GnBu",
            cbar=False,
            xticklabels=["Ham (Pred)", "Spam (Pred)"],
            yticklabels=["Ham (True)", "Spam (True)"],
            ax=ax,
            annot_kws={"size": 13, "weight": "bold"},
        )
        highlight = " (Selected Model)" if name == best_model_name else ""
        ax.set_title(f"{name} Confusion Matrix{highlight}", fontsize=11, fontweight="bold", pad=10)
        ax.set_ylabel("Actual Label", fontsize=10)
        ax.set_xlabel("Predicted Label", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=300)
    plt.close()
    logger.info("Saved reports/figures/confusion_matrix.png")

    # 3. Model Comparison Bar Chart
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    lr_scores = [
        models_results["Logistic Regression"]["accuracy"],
        models_results["Logistic Regression"]["precision"],
        models_results["Logistic Regression"]["recall"],
        models_results["Logistic Regression"]["f1_score"],
    ]
    nb_scores = [
        models_results["Multinomial Naive Bayes"]["accuracy"],
        models_results["Multinomial Naive Bayes"]["precision"],
        models_results["Multinomial Naive Bayes"]["recall"],
        models_results["Multinomial Naive Bayes"]["f1_score"],
    ]

    x = np.arange(len(metrics))
    width = 0.35

    plt.figure(figsize=(9, 5.5))
    bar1 = plt.bar(x - width / 2, nb_scores, width, label="Multinomial Naive Bayes", color="#3B82F6", edgecolor="#1E3A8A")
    bar2 = plt.bar(x + width / 2, lr_scores, width, label="Logistic Regression", color="#0D9488", edgecolor="#115E59")

    plt.title("Model Performance Comparison (Test Set)", fontsize=13, fontweight="bold", pad=15)
    plt.ylabel("Score (0.0 to 1.0)", fontsize=11)
    plt.xticks(x, metrics, fontsize=11, fontweight="medium")
    plt.ylim(0.85, 1.02)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend(frameon=True, facecolor="white", edgecolor="#CBD5E1", loc="lower right")

    for bars in [bar1, bar2]:
        for b in bars:
            val = b.get_height()
            plt.annotate(
                f"{val:.3f}",
                xy=(b.get_x() + b.get_width() / 2, val),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )
    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison.png", dpi=300)
    plt.close()
    logger.info("Saved reports/figures/model_comparison.png")

    # 4. ROC Curve Comparison Chart
    plt.figure(figsize=(7.5, 5.5))
    for name, res in models_results.items():
        fpr = res["fpr"]
        tpr = res["tpr"]
        auc = res["roc_auc"]
        color = "#0D9488" if name == "Logistic Regression" else "#3B82F6"
        plt.plot(fpr, tpr, color=color, lw=2.2, label=f"{name} (AUC = {auc:.4f})")

    plt.plot([0, 1], [0, 1], color="#94A3B8", lw=1.5, linestyle="--", label="Chance (AUC = 0.5000)")
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.03])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=11)
    plt.title("Receiver Operating Characteristic (ROC) Comparison", fontsize=13, fontweight="bold", pad=15)
    plt.legend(loc="lower right", frameon=True, facecolor="white")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_dir / "metrics_comparison.png", dpi=300)
    plt.close()
    logger.info("Saved reports/figures/metrics_comparison.png")


def train_and_evaluate_models(
    data_path: Path = Path("data/processed/emails_cleaned.csv"),
    models_dir: Path = Path("models"),
    reports_dir: Path = Path("reports"),
    random_state: int = 42,
) -> dict:
    """Train Multinomial Naive Bayes and Logistic Regression with TF-IDF,
    evaluate on test split, compare results, and persist the champion model.
    """
    if not data_path.exists():
        logger.info(f"{data_path} not found. Running data extraction first...")
        df = load_and_process_dataset()
    else:
        df = pd.read_csv(data_path)

    logger.info(f"Loaded dataset: {len(df)} total emails.")
    return run_proper_training(df, models_dir=models_dir, reports_dir=reports_dir, random_state=random_state)


def run_proper_training(
    df: pd.DataFrame,
    models_dir: Path = Path("models"),
    reports_dir: Path = Path("reports"),
    random_state: int = 42,
) -> dict:
    """Clean execution of stratified split, pipeline fitting, evaluation, and saving."""
    models_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    X = df["text"].fillna("").astype(str).tolist()
    y = np.array(df["label"].tolist(), dtype=int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=random_state
    )

    logger.info(f"Train set: {len(X_train)} samples | Test set: {len(X_test)} samples")
    logger.info(f"Train label distribution: Ham={sum(y_train == 0)}, Spam={sum(y_train == 1)}")
    logger.info(f"Test label distribution: Ham={sum(y_test == 0)}, Spam={sum(y_test == 1)}")

    # Define TF-IDF configuration with negation-preserved stop words
    tfidf = TfidfVectorizer(
        preprocessor=clean_email_text,
        stop_words=PRESERVED_STOP_WORDS,
        max_features=8000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.85,
        sublinear_tf=True,
    )

    candidates = {
        "Multinomial Naive Bayes": Pipeline([
            ("tfidf", tfidf),
            ("classifier", MultinomialNB(alpha=0.1)),
        ]),
        "Logistic Regression": Pipeline([
            ("tfidf", tfidf),
            ("classifier", LogisticRegression(
                max_iter=1000,
                C=2.0,
                random_state=random_state,
                solver="liblinear",
            )),
        ]),
    }

    models_results = {}
    fitted_pipelines = {}

    for name, pipeline in candidates.items():
        logger.info(f"Fitting pipeline for {name}...")
        pipeline.fit(X_train, y_train)
        fitted_pipelines[name] = pipeline

        # Predictions on unseen test set
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, pos_label=1))
        rec = float(recall_score(y_test, y_pred, pos_label=1))
        f1 = float(f1_score(y_test, y_pred, pos_label=1))
        auc = float(roc_auc_score(y_test, y_proba))
        cm = confusion_matrix(y_test, y_pred).tolist()
        report = classification_report(y_test, y_pred, target_names=["Ham", "Spam"], output_dict=True)

        fpr, tpr, _ = roc_curve(y_test, y_proba)

        models_results[name] = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "roc_auc": auc,
            "confusion_matrix": cm,
            "classification_report": report,
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
        }

        logger.info(f"[{name}] Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")

    # Determine Best Model based on F1-Score & Accuracy
    best_model_name = max(
        models_results.keys(),
        key=lambda k: (models_results[k]["f1_score"], models_results[k]["accuracy"]),
    )
    logger.info(f"Selected Best Model: {best_model_name} (F1: {models_results[best_model_name]['f1_score']:.4f})")

    best_pipeline = fitted_pipelines[best_model_name]

    # Save champion model
    champion_path = models_dir / "spam_classifier.joblib"
    joblib.dump(best_pipeline, champion_path)
    logger.info(f"Saved champion model to {champion_path}")

    # Also save individual models for comparison / fallback inspection
    joblib.dump(fitted_pipelines["Multinomial Naive Bayes"], models_dir / "multinomial_nb.joblib")
    joblib.dump(fitted_pipelines["Logistic Regression"], models_dir / "logistic_regression.joblib")

    # Generate evaluation figures
    generate_evaluation_figures(df, y_test, models_results, best_model_name, figures_dir)

    # Save metadata and evaluation JSON
    eval_payload = {
        "best_model": best_model_name,
        "dataset_size": len(df),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "class_distribution": {
            "ham": int((df["label"] == 0).sum()),
            "spam": int((df["label"] == 1).sum()),
        },
        "models": {
            k: {
                "accuracy": v["accuracy"],
                "precision": v["precision"],
                "recall": v["recall"],
                "f1_score": v["f1_score"],
                "roc_auc": v["roc_auc"],
                "confusion_matrix": v["confusion_matrix"],
                "classification_report": v["classification_report"],
            }
            for k, v in models_results.items()
        },
    }

    eval_json_path = reports_dir / "model_evaluation.json"
    with open(eval_json_path, "w", encoding="utf-8") as f:
        json.dump(eval_payload, f, indent=2)
    logger.info(f"Saved evaluation report to {eval_json_path}")

    return eval_payload


if __name__ == "__main__":
    df = load_and_process_dataset()
    results = run_proper_training(df)
    print("\nTraining and Evaluation Complete!")
    print(f"Champion Model: {results['best_model']}")
    print(f"Metrics: {json.dumps(results['models'][results['best_model']], indent=2)}")
