"""
src/explainability.py — Feature importance and SHAP plots
"""
import logging
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg

logger = logging.getLogger(__name__)
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Feature Importance (tree-based models)
# ─────────────────────────────────────────────────────────────────────────────
def plot_feature_importance(name: str, model, feature_names: list, top_n: int = 15, save: bool = True) -> str:
    # Unwrap Pipeline if needed
    clf = model
    if hasattr(model, "named_steps"):
        clf = model.named_steps.get("clf", model)

    if not hasattr(clf, "feature_importances_"):
        logger.warning("Model %s has no feature_importances_", name)
        return ""

    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    sorted_feats = [feature_names[i] for i in indices]
    sorted_imps  = importances[indices]

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.barh(sorted_feats[::-1], sorted_imps[::-1],
                   color=plt.cm.viridis(np.linspace(0.2, 0.85, top_n)))
    ax.set_xlabel("Importance Score")
    ax.set_title(f"Feature Importance — {name}", fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    path = ""
    if save:
        path = os.path.join(cfg.OUTPUT_DIR, f"feature_importance_{name.replace(' ', '_')}.png")
        plt.savefig(path, dpi=120, bbox_inches="tight")
        logger.info("Saved feature importance → %s", path)
    plt.close(fig)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# SHAP (optional — only if shap is installed)
# ─────────────────────────────────────────────────────────────────────────────
def plot_shap_summary(model, X_test, feature_names: list, model_name: str = "Random Forest", save: bool = True) -> str:
    try:
        import shap

        clf = model
        if hasattr(model, "named_steps"):
            clf = model.named_steps.get("clf", model)

        explainer   = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_test)

        fig, ax = plt.subplots(figsize=(10, 7))
        # shap_values is a list of arrays for multi-class
        if isinstance(shap_values, list):
            sv = np.abs(np.array(shap_values)).mean(axis=0)
        else:
            sv = np.abs(shap_values)

        mean_abs = sv.mean(axis=0)
        idx = np.argsort(mean_abs)[::-1][:15]
        ax.barh(
            [feature_names[i] for i in idx[::-1]],
            mean_abs[idx[::-1]],
            color="#6366f1"
        )
        ax.set_xlabel("|SHAP value| (mean)")
        ax.set_title(f"SHAP Feature Importance — {model_name}", fontsize=13, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()

        path = ""
        if save:
            path = os.path.join(cfg.OUTPUT_DIR, f"shap_{model_name.replace(' ', '_')}.png")
            plt.savefig(path, dpi=120, bbox_inches="tight")
            logger.info("Saved SHAP plot → %s", path)
        plt.close(fig)
        return path

    except ImportError:
        logger.info("shap not installed — skipping SHAP plot")
        return ""
    except Exception as e:
        logger.warning("SHAP plot failed: %s", e)
        return ""
