"""
src/predict.py — Single-student prediction logic
"""
import logging
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg

logger = logging.getLogger(__name__)

CLASS_DECODE = {0: "L", 1: "M", 2: "H"}
CLASS_NAMES  = {"L": "Low", "M": "Medium", "H": "High"}


# ─────────────────────────────────────────────────────────────────────────────
# Encoding helpers (loads preprocessing encoders to match training exactly)
# ─────────────────────────────────────────────────────────────────────────────
import joblib

ENCODERS_PATH = os.path.join(cfg.MODEL_DIR, "encoders.joblib")
_encoders = None

def get_encoders():
    global _encoders
    if _encoders is None:
        if os.path.exists(ENCODERS_PATH):
            try:
                _encoders = joblib.load(ENCODERS_PATH)
                logger.info("Successfully loaded encoders from %s", ENCODERS_PATH)
            except Exception as e:
                logger.error("Failed to load encoders from %s: %s", ENCODERS_PATH, e)
        else:
            logger.warning("encoders.joblib not found at %s. Predictions will fall back to static approximations.", ENCODERS_PATH)
    return _encoders


def safe_encode(col_name: str, val: str) -> int:
    encoders = get_encoders()
    if encoders and col_name in encoders:
        le = encoders[col_name]
        val_str = str(val).strip()
        if val_str in le.classes_:
            return int(le.transform([val_str])[0])
        else:
            # Case-insensitive robust fallback search
            classes_lower = [str(c).lower().strip() for c in le.classes_]
            val_lower = val_str.lower()
            if val_lower in classes_lower:
                idx = classes_lower.index(val_lower)
                return int(le.transform([le.classes_[idx]])[0])
            # Default to first category if unseen
            return 0
    else:
        # Fallback mappings in case model is untrained/missing encoders.joblib
        FALLBACK_MAPS = {
            "gender": {"M": 1, "F": 0},
            "Semester": {"F": 0, "S": 1},
            "Relation": {"Father": 0, "Mum": 1},
            "ParentAnsweringSurvey": {"Yes": 1, "No": 0},
            "ParentschoolSatisfaction": {"Good": 1, "Bad": 0},
            "StudentAbsenceDays": {"Under-7": 0, "Above-7": 1},
            "StageID": {"lowerlevel": 0, "MiddleSchool": 1, "HighSchool": 2},
            "GradeID": {f"G-{i:02d}": i for i in range(1, 13)},
        }
        val_str = str(val).strip()
        if col_name in FALLBACK_MAPS:
            return FALLBACK_MAPS[col_name].get(val_str, 0)
        # Hashing fallback for nationality, birth, section, topic
        return hash(val_str) % 30


def _clip_score(value: float) -> float:
    return float(np.clip(value, 0.0, 100.0))


def _grade_to_number(value: str) -> int:
    digits = "".join(ch for ch in str(value) if ch.isdigit())
    return int(digits) if digits else 0


def encode_input(raw: dict) -> np.ndarray:
    """
    Encode a raw dict of student attributes into a feature vector
    matching the order used in preprocessing.py.
    """
    try:
        gender_enc   = safe_encode("gender", raw.get("gender", "M"))
        nat_enc      = safe_encode("NationalITy", raw.get("NationalITy", "KW"))
        pob_enc      = safe_encode("PlaceofBirth", raw.get("PlaceofBirth", "KuwaIT"))
        stage_enc    = safe_encode("StageID", raw.get("StageID", "MiddleSchool"))
        grade_enc    = safe_encode("GradeID", raw.get("GradeID", "G-07"))
        section_enc  = safe_encode("SectionID", raw.get("SectionID", "A"))
        topic_enc    = safe_encode("Topic", raw.get("Topic", "IT"))
        semester_enc = safe_encode("Semester", raw.get("Semester", "F"))
        relation_enc = safe_encode("Relation", raw.get("Relation", "Father"))

        raised   = int(raw.get("raisedhands", 30))
        visited  = int(raw.get("VisITedResources", 30))
        announce = int(raw.get("AnnouncementsView", 15))
        discuss  = int(raw.get("Discussion", 30))

        survey_enc = safe_encode("ParentAnsweringSurvey", raw.get("ParentAnsweringSurvey", "Yes"))
        sat_enc    = safe_encode("ParentschoolSatisfaction", raw.get("ParentschoolSatisfaction", "Good"))
        absence_enc= safe_encode("StudentAbsenceDays", raw.get("StudentAbsenceDays", "Under-7"))

        # Engineered features evaluated on raw strings to match training exactly!
        engagement = (raised + visited + announce + discuss) / 4.0
        activity_total = raised + visited + announce + discuss
        active_learning = (raised + discuss) / 2.0
        digital_learning = (visited + announce) / 2.0
        communication = (announce + discuss) / 2.0
        engagement_consistency = _clip_score(100.0 - float(np.std([raised, visited, announce, discuss], ddof=1)))
        
        survey_raw = str(raw.get("ParentAnsweringSurvey", "Yes")).upper().strip()
        sat_raw    = str(raw.get("ParentschoolSatisfaction", "Good")).lower().strip()
        parent_inv = int(survey_raw == "YES") + int(sat_raw == "GOOD")
        
        absence_str = str(raw.get("StudentAbsenceDays", "Under-7")).strip()
        abs_flag    = int(absence_str.lower() == "above-7")
        support_risk = (abs_flag * 2) + (2 - parent_inv)
        engagement_trend = 0 if engagement <= 40 else 1 if engagement <= 70 else 2
        parent_support = (parent_inv / 2.0) * 100.0
        activity_consistency = engagement_consistency
        digital_ratio = ((digital_learning * 2.0 / activity_total) * 100.0) if activity_total > 0 else 0.0
        seniority = _clip_score((_grade_to_number(raw.get("GradeID", "G-07")) / 12.0) * 100.0)
        resource_utilization = _clip_score((visited * 0.7) + (announce * 0.3))
        success_index = _clip_score(
            (engagement * 0.35) +
            (active_learning * 0.20) +
            (resource_utilization * 0.20) +
            (parent_support * 0.15) +
            ((100.0 - abs_flag * 100.0) * 0.10)
        )
        academic_risk = _clip_score(
            ((100.0 - engagement) * 0.35) +
            (abs_flag * 100.0 * 0.25) +
            ((100.0 - parent_support) * 0.20) +
            ((100.0 - resource_utilization) * 0.15) +
            ((100.0 - activity_consistency) * 0.05)
        )
        active_flag = int(active_learning >= 60 and engagement >= 50)
        passive_flag = int(active_learning < 40 and engagement < 40)

        vec = np.array([
            gender_enc, nat_enc, pob_enc, stage_enc, grade_enc, section_enc, topic_enc,
            semester_enc, relation_enc, raised, visited, announce, discuss,
            survey_enc, sat_enc, absence_enc,
            engagement, parent_inv, abs_flag, activity_total, active_learning,
            digital_learning, communication, engagement_consistency, support_risk,
            engagement_trend, success_index, academic_risk, parent_support,
            activity_consistency, active_flag, passive_flag, digital_ratio,
            seniority, resource_utilization,
        ], dtype=float)
        return vec

    except Exception as e:
        logger.error("Error encoding input: %s", e)
        raise ValueError(f"Invalid input: {e}") from e


# ─────────────────────────────────────────────────────────────────────────────
# Predict with all models
# ─────────────────────────────────────────────────────────────────────────────
def predict_all(raw_input: dict, models: dict) -> list:
    """
    Returns list of dicts:
      { name, prediction (L/M/H), label, confidence }
    """
    vec = encode_input(raw_input).reshape(1, -1)
    results = []

    for name, model in models.items():
        try:
            pred_idx  = int(model.predict(vec)[0])
            pred_code = CLASS_DECODE.get(pred_idx, "M")
            pred_label = CLASS_NAMES.get(pred_code, "Medium")

            # Confidence
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(vec)[0]
                confidence = round(float(proba[pred_idx]) * 100, 1)
            else:
                confidence = None

            results.append({
                "name":       name,
                "prediction": pred_code,
                "label":      pred_label,
                "confidence": confidence,
            })
        except Exception as e:
            logger.warning("Prediction failed for %s: %s", name, e)
            results.append({"name": name, "prediction": "?", "label": "Error", "confidence": None})

    return results
