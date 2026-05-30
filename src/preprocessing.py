"""
src/preprocessing.py — Data loading, cleaning, encoding, and splitting
"""
import logging
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ── allow running from repo root ──────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load & quality report
# ─────────────────────────────────────────────────────────────────────────────
def load_data(path: str = cfg.DATA_PATH) -> pd.DataFrame:
    """Load CSV and print a basic data-quality report."""
    logger.info("Loading data from %s", path)
    df = pd.read_csv(path)
    logger.info("Dataset shape: %s", df.shape)

    nulls = df.isnull().sum()
    if nulls.any():
        logger.warning("Null values found:\n%s", nulls[nulls > 0])
        df = df.dropna()
        logger.info("Dropped rows with nulls. New shape: %s", df.shape)

    logger.info("Class distribution:\n%s", df[cfg.TARGET_COL].value_counts().to_dict())
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. Feature engineering
# ─────────────────────────────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived engagement / parent-involvement features."""
    df = df.copy()

    engagement_cols = ["raisedhands", "VisITedResources", "AnnouncementsView", "Discussion"]
    grade_num = pd.to_numeric(
        df["GradeID"].astype(str).str.extract(r"(\d+)", expand=False),
        errors="coerce",
    ).fillna(0)

    df["engagement_score"] = (
        df["raisedhands"] + df["VisITedResources"] +
        df["AnnouncementsView"] + df["Discussion"]
    ) / 4.0

    df["parent_involvement"] = (
        (df["ParentAnsweringSurvey"].str.upper() == "YES").astype(int) +
        (df["ParentschoolSatisfaction"].str.lower() == "good").astype(int)
    )

    df["absence_flag"] = (
        (df["StudentAbsenceDays"].str.lower() == "above-7").astype(int)
    )

    df["activity_total"] = df[engagement_cols].sum(axis=1)
    df["active_learning_score"] = (df["raisedhands"] + df["Discussion"]) / 2.0
    df["digital_learning_score"] = (df["VisITedResources"] + df["AnnouncementsView"]) / 2.0
    df["communication_score"] = (df["AnnouncementsView"] + df["Discussion"]) / 2.0
    df["engagement_consistency"] = (100.0 - df[engagement_cols].std(axis=1)).clip(lower=0, upper=100)
    df["support_risk_score"] = (df["absence_flag"] * 2) + (2 - df["parent_involvement"])
    df["engagement_trend_category"] = pd.cut(
        df["engagement_score"],
        bins=[-np.inf, 40, 70, np.inf],
        labels=[0, 1, 2],
    ).astype(int)
    df["parent_support_score"] = (df["parent_involvement"] / 2.0) * 100.0
    df["activity_consistency_score"] = df["engagement_consistency"]
    df["digital_engagement_ratio"] = np.where(
        df["activity_total"] > 0,
        (df["digital_learning_score"] * 2.0 / df["activity_total"]) * 100.0,
        0.0,
    )
    df["seniority_score"] = (grade_num / 12.0 * 100.0).clip(lower=0, upper=100)
    df["resource_utilization_score"] = (
        (df["VisITedResources"] * 0.7) + (df["AnnouncementsView"] * 0.3)
    ).clip(lower=0, upper=100)
    df["student_success_index"] = (
        (df["engagement_score"] * 0.35) +
        (df["active_learning_score"] * 0.20) +
        (df["resource_utilization_score"] * 0.20) +
        (df["parent_support_score"] * 0.15) +
        ((100.0 - df["absence_flag"] * 100.0) * 0.10)
    ).clip(lower=0, upper=100)
    df["academic_risk_index"] = (
        ((100.0 - df["engagement_score"]) * 0.35) +
        (df["absence_flag"] * 100.0 * 0.25) +
        ((100.0 - df["parent_support_score"]) * 0.20) +
        ((100.0 - df["resource_utilization_score"]) * 0.15) +
        ((100.0 - df["activity_consistency_score"]) * 0.05)
    ).clip(lower=0, upper=100)
    df["active_learner_flag"] = (
        (df["active_learning_score"] >= 60) & (df["engagement_score"] >= 50)
    ).astype(int)
    df["passive_learner_flag"] = (
        (df["active_learning_score"] < 40) & (df["engagement_score"] < 40)
    ).astype(int)

    logger.info("Engineered %d features: %s", len(cfg.ENGINEERED_FEATURE_COLS), cfg.ENGINEERED_FEATURE_COLS)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. Encode & prepare
# ─────────────────────────────────────────────────────────────────────────────
def encode_data(df: pd.DataFrame):
    """
    Encode all columns.
    Returns (X, y, feature_names, label_encoders, scaler)
      - X          : raw (unscaled) feature matrix (numpy)
      - y          : encoded target (numpy)
      - feature_names : list[str]
      - label_encoders: dict  { col_name -> LabelEncoder }
      - scaler     : fitted StandardScaler (on numeric-heavy cols)
    """
    df = df.copy()

    # Encode target using config CLASS_MAP to ensure L=0, M=1, H=2 order
    y = df[cfg.TARGET_COL].map(cfg.CLASS_MAP).values

    # Work on all feature columns + engineered ones
    all_feature_cols = cfg.FEATURE_COLS + cfg.ENGINEERED_FEATURE_COLS
    # Keep only cols that actually exist in the dataframe
    all_feature_cols = [c for c in all_feature_cols if c in df.columns]

    X_df = df[all_feature_cols].copy()

    label_encoders: dict = {}
    for col in cfg.CAT_COLS:
        if col in X_df.columns:
            le = LabelEncoder()
            X_df[col] = le.fit_transform(X_df[col].astype(str))
            label_encoders[col] = le

    feature_names = list(X_df.columns)
    X = X_df.values.astype(float)

    # Initialize scaler (will be fit on training set during split)
    scaler = StandardScaler()
    scaler.fit(X)

    # Save label encoders to disk
    import joblib
    try:
        os.makedirs(cfg.MODEL_DIR, exist_ok=True)
        encoders_path = os.path.join(cfg.MODEL_DIR, "encoders.joblib")
        joblib.dump(label_encoders, encoders_path)
        logger.info("Saved label encoders to %s", encoders_path)
    except Exception as e:
        logger.error("Failed to save label encoders: %s", e)

    logger.info("Encoding complete. %d features, %d samples", len(feature_names), len(y))
    return X, y, feature_names, label_encoders, scaler


# ─────────────────────────────────────────────────────────────────────────────
# 4. Train / test split
# ─────────────────────────────────────────────────────────────────────────────
def split_data(X, y):
    """Stratified 70/30 split."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg.TEST_SIZE,
        random_state=cfg.RANDOM_STATE,
        stratify=y,
    )
    logger.info(
        "Split: train=%d  test=%d  (class dist train=%s)",
        len(y_train), len(y_test),
        dict(zip(*np.unique(y_train, return_counts=True))),
    )
    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────────────────────────────────────
# 5. Full pipeline (convenience)
# ─────────────────────────────────────────────────────────────────────────────
def full_pipeline(path: str = cfg.DATA_PATH):
    """Load → engineer → encode → split.  Returns all artefacts."""
    df_raw = load_data(path)
    df_eng = engineer_features(df_raw)
    X, y, feature_names, label_encoders, scaler = encode_data(df_eng)
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Fit scaler only on X_train to prevent leakage
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    return X_train, X_test, y_train, y_test, feature_names, scaler, df_raw
