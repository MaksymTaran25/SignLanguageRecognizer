"""Evaluation utilities: metrics, confusion matrix, per-class report."""

import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

import config


def evaluate_model(pipeline, X, y, labels):
    """Print accuracy, F1, and per-class classification report."""
    y_pred = pipeline.predict(X)
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average="weighted")

    print(f"  Top-1 Accuracy: {acc:.4f}")
    print(f"  Weighted F1:    {f1:.4f}")

    # Per-class report — only include labels present in y
    present_labels = sorted(set(y))
    target_names = [config.VOCABULARY[i] for i in present_labels]
    print("\n" + classification_report(y, y_pred, labels=present_labels,
                                       target_names=target_names))
    return acc, f1


def plot_confusion_matrix(pipeline, X, y, labels, save_path=None):
    """Generate and save a confusion matrix heatmap."""
    y_pred = pipeline.predict(X)
    present_labels = sorted(set(y))
    target_names = [config.VOCABULARY[i] for i in present_labels]
    cm = confusion_matrix(y, y_pred, labels=present_labels)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=target_names, yticklabels=target_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix — Sign Language Classifier")
    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(config.MODELS_DIR, "confusion_matrix.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def find_confused_pairs(pipeline, X, y, top_n=5):
    """Find the most confused sign pairs from the confusion matrix."""
    y_pred = pipeline.predict(X)
    present_labels = sorted(set(y))
    cm = confusion_matrix(y, y_pred, labels=present_labels)

    # Zero out diagonal
    np.fill_diagonal(cm, 0)

    pairs = []
    for i in range(len(present_labels)):
        for j in range(len(present_labels)):
            if cm[i][j] > 0:
                pairs.append((
                    config.VOCABULARY[present_labels[i]],
                    config.VOCABULARY[present_labels[j]],
                    cm[i][j],
                ))

    pairs.sort(key=lambda x: x[2], reverse=True)
    if pairs:
        print(f"\nTop {min(top_n, len(pairs))} confused pairs:")
        for true_sign, pred_sign, count in pairs[:top_n]:
            print(f"  '{true_sign}' misclassified as '{pred_sign}': {count} times")
    return pairs[:top_n]
