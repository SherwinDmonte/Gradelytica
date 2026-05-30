"""
src/visualize.py — EDA visualisation helpers (static matplotlib/seaborn)
"""
import logging
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg

logger = logging.getLogger(__name__)
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")


def _save(fig, filename: str) -> str:
    path = os.path.join(cfg.OUTPUT_DIR, filename)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    logger.info("Saved chart → %s", path)
    plt.close(fig)
    return path


def plot_class_distribution(df):
    fig, ax = plt.subplots(figsize=(7, 4))
    order = ["L", "M", "H"]
    colors = [cfg.CLASS_COLORS[c] for c in order]
    sns.countplot(x="Class", data=df, order=order, palette=colors, ax=ax)
    ax.set_title("Student Performance Class Distribution", fontweight="bold")
    ax.set_xlabel("Class (L=Low, M=Medium, H=High)")
    return _save(fig, "eda_class_distribution.png")


def plot_correlation_heatmap(df):
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm",
                linewidths=0.5, ax=ax)
    ax.set_title("Feature Correlation Heatmap", fontweight="bold")
    return _save(fig, "eda_correlation_heatmap.png")


def plot_semester_class(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(x="Semester", hue="Class", data=df, hue_order=["L", "M", "H"],
                  palette=["#ef4444", "#f59e0b", "#22c55e"], ax=ax)
    ax.set_title("Class Distribution by Semester", fontweight="bold")
    return _save(fig, "eda_semester_class.png")


def plot_gender_class(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(x="gender", hue="Class", data=df, order=["M", "F"],
                  hue_order=["L", "M", "H"],
                  palette=["#ef4444", "#f59e0b", "#22c55e"], ax=ax)
    ax.set_title("Class Distribution by Gender", fontweight="bold")
    return _save(fig, "eda_gender_class.png")


def plot_absence_class(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(x="StudentAbsenceDays", hue="Class", data=df,
                  hue_order=["L", "M", "H"],
                  palette=["#ef4444", "#f59e0b", "#22c55e"], ax=ax)
    ax.set_title("Class Distribution by Absence Days", fontweight="bold")
    return _save(fig, "eda_absence_class.png")


def plot_engagement_boxplot(df):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    cols = ["raisedhands", "VisITedResources", "AnnouncementsView", "Discussion"]
    order = ["L", "M", "H"]
    for ax, col in zip(axes.flat, cols):
        sns.boxplot(x="Class", y=col, data=df, order=order,
                    palette=["#ef4444", "#f59e0b", "#22c55e"], ax=ax)
        ax.set_title(col, fontweight="bold")
    fig.suptitle("Engagement Metrics by Class", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return _save(fig, "eda_engagement_boxplot.png")


def plot_topic_class(df):
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.countplot(x="Topic", hue="Class", data=df, hue_order=["L", "M", "H"],
                  palette=["#ef4444", "#f59e0b", "#22c55e"], ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    ax.set_title("Class Distribution by Topic", fontweight="bold")
    return _save(fig, "eda_topic_class.png")


def plot_nationality_class(df):
    fig, ax = plt.subplots(figsize=(14, 5))
    sns.countplot(x="NationalITy", hue="Class", data=df, hue_order=["L", "M", "H"],
                  palette=["#ef4444", "#f59e0b", "#22c55e"], ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
    ax.set_title("Class Distribution by Nationality", fontweight="bold")
    return _save(fig, "eda_nationality_class.png")


def plot_stage_class(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(x="StageID", hue="Class", data=df, hue_order=["L", "M", "H"],
                  palette=["#ef4444", "#f59e0b", "#22c55e"], ax=ax)
    ax.set_title("Class Distribution by School Stage", fontweight="bold")
    return _save(fig, "eda_stage_class.png")


def generate_all_eda(df) -> list:
    """Run all EDA plots and return list of saved file paths."""
    plots = [
        plot_class_distribution(df),
        plot_correlation_heatmap(df),
        plot_semester_class(df),
        plot_gender_class(df),
        plot_absence_class(df),
        plot_engagement_boxplot(df),
        plot_topic_class(df),
        plot_nationality_class(df),
        plot_stage_class(df),
    ]
    logger.info("Generated %d EDA charts.", len(plots))
    return plots
