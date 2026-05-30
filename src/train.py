"""
src/train.py — Model training pipeline with cross-validation and hyperparameter tuning
"""
import logging
import os
import sys
import time

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg

logger = logging.getLogger(__name__)

os.makedirs(cfg.MODEL_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: build sklearn estimator (with or without scaling pipeline)
# ─────────────────────────────────────────────────────────────────────────────
def _build_estimator(name: str, params: dict):
    MODEL_CLS = {
        "Decision Tree":      DecisionTreeClassifier,
        "Random Forest":      RandomForestClassifier,
        "Gradient Boosting":  GradientBoostingClassifier,
        "SVM":                SVC,
        "Logistic Regression":LogisticRegression,
        "Perceptron":         Perceptron,
        "MLP Classifier":     MLPClassifier,
    }
    clf = MODEL_CLS[name](**params)
    if name in cfg.SCALED_MODELS:
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    return clf


# ─────────────────────────────────────────────────────────────────────────────
# Cross-validate a single model
# ─────────────────────────────────────────────────────────────────────────────
def cross_validate_model(name, estimator, X, y) -> dict:
    cv = StratifiedKFold(n_splits=cfg.CV_FOLDS, shuffle=True, random_state=cfg.RANDOM_STATE)
    acc_scores = cross_val_score(estimator, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    f1_scores  = cross_val_score(estimator, X, y, cv=cv, scoring="f1_weighted", n_jobs=-1)
    result = {
        "cv_accuracy_mean": round(float(acc_scores.mean()), 4),
        "cv_accuracy_std":  round(float(acc_scores.std()),  4),
        "cv_f1_mean":       round(float(f1_scores.mean()),  4),
        "cv_f1_std":        round(float(f1_scores.std()),   4),
    }
    logger.info(
        "[CV] %s  acc=%.3f±%.3f  f1=%.3f±%.3f",
        name, result["cv_accuracy_mean"], result["cv_accuracy_std"],
        result["cv_f1_mean"], result["cv_f1_std"],
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameter tuning
# ─────────────────────────────────────────────────────────────────────────────
def tune_model(name: str, X_train, y_train):
    """Run RandomizedSearchCV for models listed in cfg.TUNING_GRIDS."""
    if name not in cfg.TUNING_GRIDS:
        return None

    logger.info("Hyperparameter tuning for %s ...", name)
    base_params = dict(cfg.MODEL_PARAMS[name])
    base_params.pop("random_state", None)   # will be set in grid
    estimator = _build_estimator(name, {})  # bare estimator for tuning

    param_grid = cfg.TUNING_GRIDS[name]
    if name in cfg.SCALED_MODELS:
        param_grid = {f"clf__{k}": v for k, v in param_grid.items()}

    search = RandomizedSearchCV(
        estimator,
        param_distributions=param_grid,
        n_iter=20,
        cv=StratifiedKFold(n_splits=cfg.CV_FOLDS, shuffle=True, random_state=cfg.RANDOM_STATE),
        scoring="accuracy",
        random_state=cfg.RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_train, y_train)
    logger.info("Best params for %s: %s  (score=%.4f)", name, search.best_params_, search.best_score_)
    return search.best_estimator_


# ─────────────────────────────────────────────────────────────────────────────
# Train all models
# ─────────────────────────────────────────────────────────────────────────────
def train_all(X_train, y_train, X_full=None, y_full=None, tune: bool = False):
    """
    Train all 7 models.
    Returns dict { name -> {"model": fitted_estimator, "cv": {...}, "train_time": float} }
    """
    results = {}

    for name, params in cfg.MODEL_PARAMS.items():
        logger.info("─── Training %s ───", name)
        t0 = time.time()

        # Optionally tune top 2 tree-based models
        if tune and name in cfg.TUNING_GRIDS:
            model = tune_model(name, X_train, y_train)
            if model is None:
                model = _build_estimator(name, params)
                model.fit(X_train, y_train)
        else:
            model = _build_estimator(name, params)
            model.fit(X_train, y_train)

        elapsed = round(time.time() - t0, 2)

        # Cross-val on full available data (or training set if X_full not provided)
        X_cv = X_full if X_full is not None else X_train
        y_cv = y_full if y_full is not None else y_train
        cv_stats = cross_validate_model(name, _build_estimator(name, params), X_cv, y_cv)

        results[name] = {
            "model":      model,
            "cv":         cv_stats,
            "train_time": elapsed,
        }

        # Persist to disk
        model_path = os.path.join(cfg.MODEL_DIR, f"{name.replace(' ', '_')}.pkl")
        joblib.dump(model, model_path)
        logger.info("Saved model → %s  (%.2fs)", model_path, elapsed)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Load persisted models (skip re-training)
# ─────────────────────────────────────────────────────────────────────────────
def load_models() -> dict:
    """Load all previously saved .pkl models from MODEL_DIR."""
    models = {}
    for name in cfg.MODEL_PARAMS:
        path = os.path.join(cfg.MODEL_DIR, f"{name.replace(' ', '_')}.pkl")
        if os.path.exists(path):
            models[name] = joblib.load(path)
            logger.info("Loaded model from disk: %s", name)
    return models
