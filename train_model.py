"""
train_model.py
Trains a TF-IDF + Logistic Regression pipeline for movie genre classification.
Saves the model and prints a full evaluation report.
"""

import pandas as pd
import numpy as np
import pickle
import json
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score
)
from sklearn.preprocessing import LabelEncoder

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

CLEAN_FILE  = "data/movies_clean.csv"
MODEL_FILE  = "models/genre_classifier.pkl"
ENCODER_FILE = "models/label_encoder.pkl"
METRICS_FILE = "models/metrics.json"


def load_data(path: str):
    df = pd.read_csv(path)
    X = df["clean_text"].astype(str)
    y = df["genre"]
    print(f"Loaded {len(df)} samples across {y.nunique()} genres")
    return X, y, df


def build_pipeline() -> Pipeline:
    """TF-IDF vectoriser + Logistic Regression."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),     # unigrams + bigrams
            max_features=5000,
            min_df=1,
            sublinear_tf=True,      # apply log(1+tf) scaling
        )),
        ("clf", LogisticRegression(
            C=1.0,
            max_iter=1000,
            solver="lbfgs",
            random_state=42,
        )),
    ])


def plot_confusion_matrix(cm_arr, classes, path="outputs/3_confusion_matrix.png"):
    fig, ax = plt.subplots(figsize=(10, 8), facecolor="#F8F8F8")
    im = ax.imshow(cm_arr, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(classes, fontsize=9)

    thresh = cm_arr.max() / 2.0
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, str(cm_arr[i, j]),
                    ha="center", va="center", fontsize=11, fontweight="bold",
                    color="white" if cm_arr[i, j] > thresh else "black")

    ax.set_title("Confusion Matrix — Test Set", fontsize=14, fontweight="bold")
    ax.set_ylabel("True Genre", fontsize=11)
    ax.set_xlabel("Predicted Genre", fontsize=11)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {path}")


def plot_cv_scores(cv_scores, path="outputs/4_cv_scores.png"):
    fig, ax = plt.subplots(figsize=(8, 4), facecolor="#F8F8F8")
    folds = [f"Fold {i+1}" for i in range(len(cv_scores))]
    bars = ax.bar(folds, cv_scores, color="#1565C0", alpha=0.8, edgecolor="white")
    ax.axhline(cv_scores.mean(), color="red", linestyle="--", linewidth=1.5,
               label=f"Mean = {cv_scores.mean():.3f}")
    ax.set_ylim(0, 1.05)
    ax.set_title("5-Fold Stratified Cross-Validation Accuracy", fontsize=13, fontweight="bold")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.set_facecolor("white")
    for bar, val in zip(bars, cv_scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {path}")


def plot_per_class_f1(report_dict, path="outputs/5_per_class_f1.png"):
    genres = [k for k in report_dict if k not in ("accuracy", "macro avg", "weighted avg")]
    f1s    = [report_dict[g]["f1-score"] for g in genres]
    colors = cm.tab10(np.linspace(0, 1, len(genres)))

    fig, ax = plt.subplots(figsize=(10, 5), facecolor="#F8F8F8")
    bars = ax.bar(genres, f1s, color=colors, alpha=0.85, edgecolor="white")
    ax.set_ylim(0, 1.1)
    ax.set_title("F1-Score per Genre", fontsize=13, fontweight="bold")
    ax.set_ylabel("F1-Score")
    ax.set_facecolor("white")
    plt.xticks(rotation=30, ha="right")
    for bar, val in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.2f}", ha="center", fontsize=9, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {path}")


def main():
    X, y, df = load_data(CLEAN_FILE)

    # Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # Train / test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )
    print(f"Train: {len(X_train)}  |  Test: {len(X_test)}")

    # Build & train pipeline
    pipeline = build_pipeline()
    print("\nTraining TF-IDF + Logistic Regression pipeline ...")
    pipeline.fit(X_train, y_train)

    # --- Cross-validation ---
    print("\nRunning 5-fold stratified cross-validation ...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X, y_enc, cv=cv, scoring="accuracy")
    print(f"  CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    plot_cv_scores(cv_scores)

    # --- Test evaluation ---
    y_pred = pipeline.predict(X_test)
    acc  = accuracy_score(y_test, y_pred)
    f1_w = f1_score(y_test, y_pred, average="weighted")
    f1_m = f1_score(y_test, y_pred, average="macro")

    print(f"\n{'='*55}")
    print(f"  Test Accuracy          : {acc:.4f}  ({acc*100:.1f}%)")
    print(f"  Weighted F1-Score      : {f1_w:.4f}")
    print(f"  Macro F1-Score         : {f1_m:.4f}")
    print(f"{'='*55}")

    class_names = le.classes_
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    print(f"\nPer-class report:\n{classification_report(y_test, y_pred, target_names=class_names)}")

    # --- Plots ---
    cm_arr = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm_arr, class_names)
    plot_per_class_f1(report)

    # --- Top TF-IDF features per genre ---
    print("\nTop 5 TF-IDF keywords per genre:")
    tfidf   = pipeline.named_steps["tfidf"]
    clf     = pipeline.named_steps["clf"]
    feature_names = tfidf.get_feature_names_out()
    for i, genre in enumerate(class_names):
        top_idx = clf.coef_[i].argsort()[-5:][::-1]
        top_words = [feature_names[j] for j in top_idx]
        print(f"  {genre:<25}: {', '.join(top_words)}")

    # --- Save model, encoder, metrics ---
    with open(MODEL_FILE,   "wb") as f: pickle.dump(pipeline, f)
    with open(ENCODER_FILE, "wb") as f: pickle.dump(le, f)

    metrics = {
        "test_accuracy":    round(acc, 4),
        "weighted_f1":      round(f1_w, 4),
        "macro_f1":         round(f1_m, 4),
        "cv_accuracy_mean": round(float(cv_scores.mean()), 4),
        "cv_accuracy_std":  round(float(cv_scores.std()), 4),
        "train_size":       len(X_train),
        "test_size":        len(X_test),
        "genres":           list(class_names),
    }
    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nModel saved   → {MODEL_FILE}")
    print(f"Encoder saved → {ENCODER_FILE}")
    print(f"Metrics saved → {METRICS_FILE}")


if __name__ == "__main__":
    main()
