"""
cli.py — Clean command-line interface (preserves original functionality + upgrades)
"""
import logging
import os
import sys
import time

import numpy as np

# ── Setup logging ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg
from src.preprocessing import full_pipeline
from src.train import load_models, train_all
from src.evaluate import (
    compute_metrics,
    plot_confusion_matrix,
    plot_learning_curve,
    plot_model_comparison,
    plot_roc_curves,
)
from src.explainability import plot_feature_importance, plot_shap_summary
from src.visualize import generate_all_eda
from src.predict import predict_all

DIVIDER = "─" * 60


def print_banner():
    print("\n" + "═" * 60)
    print("   🎓  Student Performance Prediction — Advanced ML")
    print("═" * 60 + "\n")


def show_eda_menu(df_raw):
    while True:
        print("\n" + DIVIDER)
        print(" EDA / Visualisation Menu")
        print(DIVIDER)
        print(" 1. Class count distribution")
        print(" 2. Semester-wise class chart")
        print(" 3. Gender-wise class chart")
        print(" 4. Absence-wise class chart")
        print(" 5. Engagement box-plots (all 4 metrics)")
        print(" 6. Topic-wise class chart")
        print(" 7. Nationality-wise class chart")
        print(" 8. Stage-wise class chart")
        print(" 9. Correlation heatmap")
        print("10. Generate ALL charts (saves to outputs/)")
        print(" 0. Back to main menu")
        print(DIVIDER)

        try:
            ch = int(input("Choice: "))
        except ValueError:
            continue

        from src.visualize import (
            plot_absence_class, plot_class_distribution,
            plot_correlation_heatmap, plot_engagement_boxplot,
            plot_gender_class, plot_nationality_class,
            plot_semester_class, plot_stage_class, plot_topic_class,
            generate_all_eda,
        )

        chart_map = {
            1: plot_class_distribution,
            2: plot_semester_class,
            3: plot_gender_class,
            4: plot_absence_class,
            5: plot_engagement_boxplot,
            6: plot_topic_class,
            7: plot_nationality_class,
            8: plot_stage_class,
            9: plot_correlation_heatmap,
        }

        if ch == 0:
            break
        elif ch == 10:
            paths = generate_all_eda(df_raw)
            print(f"\n✅  {len(paths)} charts saved to outputs/")
        elif ch in chart_map:
            path = chart_map[ch](df_raw)
            print(f"✅  Chart saved → {path}")
        else:
            print("Invalid choice.")


def run_prediction(models):
    print("\n" + DIVIDER)
    print(" Single-Student Prediction")
    print(DIVIDER)
    raw = {}

    raw["gender"]    = input(" Gender (M / F): ").strip()
    raw["NationalITy"] = input(" Nationality (e.g. KW / Jordan / Egypt): ").strip()
    raw["PlaceofBirth"] = input(" Place of Birth: ").strip()

    stage_opts = {"1": "lowerlevel", "2": "MiddleSchool", "3": "HighSchool"}
    print(" Stage: 1=lowerlevel  2=MiddleSchool  3=HighSchool")
    raw["StageID"] = stage_opts.get(input(" Choice: ").strip(), "MiddleSchool")

    grade = input(" Grade ID (e.g. G-07): ").strip()
    raw["GradeID"] = grade if grade.startswith("G-") else "G-07"

    raw["SectionID"] = input(" Section (A / B / C): ").strip()
    raw["Topic"]     = input(" Topic (e.g. IT / Math / Science): ").strip()
    raw["Semester"]  = input(" Semester (F / S): ").strip()
    raw["Relation"]  = input(" Relation (Father / Mum): ").strip()

    try:
        raw["raisedhands"]      = int(input(" Raised hands (0-100): "))
        raw["VisITedResources"] = int(input(" Visited resources (0-100): "))
        raw["AnnouncementsView"]= int(input(" Announcements viewed (0-100): "))
        raw["Discussion"]       = int(input(" Discussions (0-100): "))
    except ValueError:
        print("⚠️  Non-numeric value entered — defaulting to 30.")
        raw.setdefault("raisedhands", 30)
        raw.setdefault("VisITedResources", 30)
        raw.setdefault("AnnouncementsView", 15)
        raw.setdefault("Discussion", 30)

    raw["ParentAnsweringSurvey"]   = input(" Parent answered survey (Yes / No): ").strip()
    raw["ParentschoolSatisfaction"]= input(" Parent satisfaction (Good / Bad): ").strip()
    raw["StudentAbsenceDays"]      = input(" Absence days (Under-7 / Above-7): ").strip()

    print("\n⏳  Running prediction across all models...\n")
    results = predict_all(raw, models)

    EMOJI = {"L": "🔴", "M": "🟡", "H": "🟢"}
    for r in results:
        conf_str = f"  (confidence: {r['confidence']}%)" if r["confidence"] else ""
        print(f"  {EMOJI.get(r['prediction'], '⚪')} {r['name']:<25} → {r['label']}{conf_str}")

    print()


def main():
    print_banner()
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    os.makedirs(cfg.MODEL_DIR, exist_ok=True)

    # ── Load / train ──────────────────────────────────────────────────────────
    logger.info("Running full preprocessing pipeline...")
    X_train, X_test, y_train, y_test, feature_names, scaler, df_raw = full_pipeline()

    saved = load_models()
    if len(saved) == len(cfg.MODEL_PARAMS):
        logger.info("All models loaded from disk — skipping retraining.")
        train_results = {name: {"model": m, "cv": {"cv_accuracy_mean": 0, "cv_accuracy_std": 0,
                                                     "cv_f1_mean": 0, "cv_f1_std": 0},
                                "train_time": 0}
                         for name, m in saved.items()}
    else:
        tune = input("\n🔧 Run hyperparameter tuning? (adds ~60s) [y/N]: ").strip().lower() == "y"
        X_full = np.vstack([X_train, X_test])
        y_full = np.concatenate([y_train, y_test])
        train_results = train_all(X_train, y_train, X_full, y_full, tune=tune)

    models = {name: r["model"] for name, r in train_results.items()}

    # ── Evaluate all models ───────────────────────────────────────────────────
    print(f"\n{DIVIDER}")
    print(" Model Evaluation Results")
    print(DIVIDER)

    metrics_list = []
    for name, r in train_results.items():
        m = compute_metrics(name, r["model"], X_test, y_test, r["cv"])
        metrics_list.append(m)

        print(f"\n{'─'*40}")
        print(f" {name}")
        print(f"{'─'*40}")
        print(f"  Accuracy : {m['accuracy']:.4f}")
        print(f"  F1-Score : {m['f1']:.4f}")
        print(f"  AUC-ROC  : {m['auc_roc'] or 'N/A'}")
        print(f"  CV Acc   : {m['cv_accuracy']}")
        print()
        print(m["report"])
        time.sleep(0.3)

    # ── Save charts ───────────────────────────────────────────────────────────
    print(f"\n{DIVIDER}\n Generating evaluation charts...\n{DIVIDER}")

    for m_data in metrics_list:
        plot_confusion_matrix(m_data["name"], y_test, m_data["y_pred"])

    plot_roc_curves(metrics_list, y_test)
    plot_model_comparison(metrics_list)

    # Best model → learning curve + feature importance
    best = max(metrics_list, key=lambda x: x["accuracy"])
    best_model = models[best["name"]]
    print(f"\n🏆 Best model: {best['name']} (accuracy={best['accuracy']:.4f})")

    X_full = np.vstack([X_train, X_test])
    y_full = np.concatenate([y_train, y_test])
    plot_learning_curve(best["name"], best_model, X_full, y_full)

    for name in ["Decision Tree", "Random Forest", "Gradient Boosting"]:
        if name in models:
            plot_feature_importance(name, models[name], feature_names)

    plot_shap_summary(models.get("Random Forest", best_model), X_test, feature_names)

    print(f"\n✅  All charts saved to → {cfg.OUTPUT_DIR}\n")

    # ── Main menu ─────────────────────────────────────────────────────────────
    while True:
        print(f"\n{DIVIDER}")
        print(" Main Menu")
        print(DIVIDER)
        print(" 1. EDA / Visualisation charts")
        print(" 2. Predict for a student")
        print(" 3. Re-evaluate models")
        print(" 0. Exit")
        print(DIVIDER)

        try:
            ch = int(input("Choice: "))
        except ValueError:
            continue

        if ch == 0:
            print("\n👋  Goodbye!\n")
            break
        elif ch == 1:
            show_eda_menu(df_raw)
        elif ch == 2:
            run_prediction(models)
        elif ch == 3:
            for m_data in metrics_list:
                print(f"  {m_data['name']:<25} acc={m_data['accuracy']:.4f}  f1={m_data['f1']:.4f}  auc={m_data['auc_roc'] or 'N/A'}")


if __name__ == "__main__":
    main()
