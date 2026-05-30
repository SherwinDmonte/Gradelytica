"""
config.py — Central configuration for Student Performance Prediction ML
"""
import os

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "data", "AI-Data.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# ──────────────────────────────────────────────
# Data / Training
# ──────────────────────────────────────────────
TARGET_COL   = "Class"
TEST_SIZE    = 0.30
RANDOM_STATE = 42
CV_FOLDS     = 5

# Columns to keep (all 16 features)
FEATURE_COLS = [
    "gender", "NationalITy", "PlaceofBirth", "StageID", "GradeID",
    "SectionID", "Topic", "Semester", "Relation",
    "raisedhands", "VisITedResources", "AnnouncementsView", "Discussion",
    "ParentAnsweringSurvey", "ParentschoolSatisfaction", "StudentAbsenceDays",
]

# Categorical columns (need encoding)
CAT_COLS = [
    "gender", "NationalITy", "PlaceofBirth", "StageID", "GradeID",
    "SectionID", "Topic", "Semester", "Relation",
    "ParentAnsweringSurvey", "ParentschoolSatisfaction", "StudentAbsenceDays",
]

# Numeric columns (no encoding needed)
NUM_COLS = ["raisedhands", "VisITedResources", "AnnouncementsView", "Discussion"]

# Engineered columns added in src.preprocessing.engineer_features
ENGINEERED_FEATURE_COLS = [
    "engagement_score",
    "parent_involvement",
    "absence_flag",
    "activity_total",
    "active_learning_score",
    "digital_learning_score",
    "communication_score",
    "engagement_consistency",
    "support_risk_score",
    "engagement_trend_category",
    "student_success_index",
    "academic_risk_index",
    "parent_support_score",
    "activity_consistency_score",
    "active_learner_flag",
    "passive_learner_flag",
    "digital_engagement_ratio",
    "seniority_score",
    "resource_utilization_score",
]

# Class label mapping
CLASS_MAP = {"L": 0, "M": 1, "H": 2}
CLASS_LABELS = ["Low (L)", "Medium (M)", "High (H)"]
CLASS_COLORS = {"L": "#ef4444", "M": "#f59e0b", "H": "#22c55e"}

# ──────────────────────────────────────────────
# Model Hyperparameters
# ──────────────────────────────────────────────
MODEL_PARAMS = {
    "Decision Tree": {
        "max_depth": None,
        "min_samples_split": 2,
        "random_state": RANDOM_STATE,
    },
    "Random Forest": {
        "n_estimators": 100,
        "max_depth": None,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    },
    "Gradient Boosting": {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 3,
        "random_state": RANDOM_STATE,
    },
    "SVM": {
        "kernel": "rbf",
        "probability": True,
        "random_state": RANDOM_STATE,
    },
    "Logistic Regression": {
        "max_iter": 1000,
        "random_state": RANDOM_STATE,
    },
    "Perceptron": {
        "random_state": RANDOM_STATE,
        "max_iter": 1000,
    },
    "MLP Classifier": {
        "activation": "logistic",
        "hidden_layer_sizes": (100,),
        "max_iter": 1000,
        "random_state": RANDOM_STATE,
    },
}

# Models that need feature scaling
SCALED_MODELS = {"Logistic Regression", "Perceptron", "SVM", "MLP Classifier"}

# ──────────────────────────────────────────────
# Hyperparameter Tuning Grids
# ──────────────────────────────────────────────
TUNING_GRIDS = {
    "Random Forest": {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 5, 10, 15],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    },
    "Gradient Boosting": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.05, 0.1, 0.2],
        "max_depth": [2, 3, 4],
        "subsample": [0.8, 1.0],
    },
}
