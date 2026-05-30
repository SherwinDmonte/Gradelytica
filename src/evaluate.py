"""
src/evaluate.py — Metrics, confusion matrix, ROC, learning curves
"""
import logging
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import label_binarize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg

logger = logging.getLogger(__name__)
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Per-model metrics
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(name: str, model, X_test, y_test, cv_stats: dict) -> dict:
    y_pred = model.predict(X_test)

    # AUC (one-vs-rest, macro)
    try:
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)
        else:
            y_proba = None
        y_bin = label_binarize(y_test, classes=[0, 1, 2])
        auc = round(roc_auc_score(y_bin, y_proba, multi_class="ovr", average="macro"), 4) if y_proba is not None else None
    except Exception:
        auc = None

    metrics = {
        "name":       name,
        "accuracy":   round(accuracy_score(y_test, y_pred), 4),
        "precision":  round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "recall":     round(recall_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "f1":         round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "auc_roc":    auc,
        "cv_accuracy": f"{cv_stats['cv_accuracy_mean']:.3f} ± {cv_stats['cv_accuracy_std']:.3f}",
        "cv_f1":       f"{cv_stats['cv_f1_mean']:.3f} ± {cv_stats['cv_f1_std']:.3f}",
        "report":      classification_report(y_test, y_pred, target_names=["Low", "Medium", "High"], zero_division=0),
        "y_pred":      y_pred,
        "y_proba":     y_proba,
    }
    logger.info("[Eval] %s  acc=%.3f  f1=%.3f  auc=%s", name, metrics["accuracy"], metrics["f1"], auc)
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Confusion matrix
# ─────────────────────────────────────────────────────────────────────────────
def plot_confusion_matrix(name: str, y_test, y_pred, save: bool = True) -> str:
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(["Low", "Medium", "High"])
    ax.set_yticklabels(["Low", "Medium", "High"])
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(f"Confusion Matrix — {name}", fontsize=13, fontweight="bold")

    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)

    plt.tight_layout()
    path = ""
    if save:
        path = os.path.join(cfg.OUTPUT_DIR, f"confusion_matrix_{name.replace(' ', '_')}.png")
        plt.savefig(path, dpi=120, bbox_inches="tight")
        logger.info("Saved confusion matrix → %s", path)
    plt.close(fig)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# ROC curves (multi-class OvR)
# ─────────────────────────────────────────────────────────────────────────────
def plot_roc_curves(metrics_list: list, y_test, save: bool = True) -> str:
    classes = [0, 1, 2]
    class_labels = ["Low", "Medium", "High"]
    colors = ["#ef4444", "#f59e0b", "#22c55e"]
    linestyles = ["-", "--", ":"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("ROC Curves — All Models (One-vs-Rest per Class)", fontsize=13, fontweight="bold")

    for ax, cls, label, color in zip(axes, classes, class_labels, colors):
        y_bin = (y_test == cls).astype(int)
        for m_data in metrics_list:
            if m_data["y_proba"] is None:
                continue
            y_score = m_data["y_proba"][:, cls]
            fpr, tpr, _ = roc_curve(y_bin, y_score)
            auc_val = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f"{m_data['name']} (AUC={auc_val:.2f})", linewidth=1.5)

        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"Class: {label}", fontweight="bold")
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(alpha=0.3)

    plt.tight_layout()
    path = ""
    if save:
        path = os.path.join(cfg.OUTPUT_DIR, "roc_curves.png")
        plt.savefig(path, dpi=120, bbox_inches="tight")
        logger.info("Saved ROC curves → %s", path)
    plt.close(fig)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Learning curve
# ─────────────────────────────────────────────────────────────────────────────
def plot_learning_curve(name: str, model, X, y, save: bool = True) -> str:
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=5, scoring="accuracy",
        train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1,
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(train_sizes,
                    train_scores.mean(1) - train_scores.std(1),
                    train_scores.mean(1) + train_scores.std(1), alpha=0.15, color="#3b82f6")
    ax.fill_between(train_sizes,
                    val_scores.mean(1) - val_scores.std(1),
                    val_scores.mean(1) + val_scores.std(1), alpha=0.15, color="#22c55e")
    ax.plot(train_sizes, train_scores.mean(1), "o-", color="#3b82f6", label="Training score")
    ax.plot(train_sizes, val_scores.mean(1), "s-",  color="#22c55e", label="CV score")
    ax.set_xlabel("Training set size")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Learning Curve — {name}", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    path = ""
    if save:
        path = os.path.join(cfg.OUTPUT_DIR, f"learning_curve_{name.replace(' ', '_')}.png")
        plt.savefig(path, dpi=120, bbox_inches="tight")
        logger.info("Saved learning curve → %s", path)
    plt.close(fig)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Model comparison bar chart
# ─────────────────────────────────────────────────────────────────────────────
def plot_model_comparison(metrics_list: list, save: bool = True) -> str:
    names = [m["name"] for m in metrics_list]
    accs  = [m["accuracy"] for m in metrics_list]
    f1s   = [m["f1"] for m in metrics_list]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width / 2, accs, width, label="Accuracy", color="#3b82f6", alpha=0.85)
    bars2 = ax.bar(x + width / 2, f1s,  width, label="F1-Score", color="#22c55e", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — Accuracy & F1-Score", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    path = ""
    if save:
        path = os.path.join(cfg.OUTPUT_DIR, "model_comparison.png")
        plt.savefig(path, dpi=120, bbox_inches="tight")
        logger.info("Saved model comparison → %s", path)
    plt.close(fig)
    return path
