"""
app.py — Premium Responsive Multi-Page Dashboard (Light & Dark Theme Adaptive)
Run with:  streamlit run app.py
"""
import os
import sys
import logging

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg
from src.preprocessing import full_pipeline, load_data, engineer_features
from src.train import train_all, load_models
from src.evaluate import compute_metrics
from src.predict import predict_all, CLASS_DECODE

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ─────────────────────────────────────────────────────────────────────────────
# 1. Page Configuration & Adaptive Multi-Theme CSS Styling
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Gradelytica — Predictive Student Analytics",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Streamlit-native variable styling for seamless Light/Dark theme switching
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    /* Core Typography overrides */
    html, body, [class*="css"], .stMarkdown, p, div, label {
        font-family: 'Outfit', sans-serif !important;
    }
    
    /* Adaptive Glassmorphic Stat Cards (light/dark adaptive) */
    .stat-card {
        background-color: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.15);
        border-radius: 12px;
        padding: 24px 20px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
        color: var(--text-color) !important;
    }
    .stat-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(128, 128, 128, 0.15);
        border-color: rgba(128, 128, 128, 0.3);
    }
    .stat-card h3 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 700;
        color: #2563eb; /* Primary royal blue for emphasis */
        letter-spacing: -0.5px;
    }
    .stat-card p {
        margin: 6px 0 0;
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        color: var(--text-color);
        opacity: 0.7;
        letter-spacing: 0.8px;
    }
    
    /* Adaptive Prediction Cards */
    .pred-card {
        background-color: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.12);
        border-radius: 10px;
        padding: 18px;
        margin-bottom: 12px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.03);
        color: var(--text-color) !important;
        transition: border-color 0.2s ease;
    }
    .pred-card:hover {
        border-color: rgba(128, 128, 128, 0.25);
    }
    
    /* Custom Badges */
    .badge {
        display: inline-block;
        padding: 4px 10px;
        font-size: 0.76rem;
        font-weight: 700;
        border-radius: 6px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .badge-high {
        background-color: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .badge-medium {
        background-color: rgba(245, 158, 11, 0.15);
        color: #f59e0b;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    .badge-low {
        background-color: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    
    /* Clean Custom Report Progress Bars */
    .progress-container {
        width: 100%;
        background-color: rgba(128, 128, 128, 0.15);
        border-radius: 4px;
        margin-top: 8px;
        height: 6px;
        overflow: hidden;
    }
    .progress-bar-fill {
        height: 100%;
        border-radius: 4px;
    }

    /* Native Sidebar integration without forced colors (adaptive to light/dark) */
    div[data-testid="stSidebar"] {
        border-right: 1px solid rgba(128, 128, 128, 0.1);
    }

    /* Hide Streamlit Deploy Button */
    .stDeployButton {
        display: none !important;
    }

    /* Clean Streamlit form styling */
    div[data-testid="stForm"] {
        background-color: var(--background-color);
        border: 1px solid rgba(128, 128, 128, 0.15) !important;
        border-radius: 12px;
        padding: 24px;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Cached ML Pipeline & Model Loader
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading student analytics dataset ...")
def get_data():
    return load_data()

@st.cache_resource(show_spinner="Initializing predictive modeling pipelines ...")
def get_pipeline_and_models():
    X_tr, X_te, y_tr, y_te, feat_names, scaler, df_raw = full_pipeline()
    saved = load_models()
    if len(saved) == len(cfg.MODEL_PARAMS):
        models = saved
        cv_stub = {"cv_accuracy_mean": 0, "cv_accuracy_std": 0,
                   "cv_f1_mean": 0, "cv_f1_std": 0}
        train_results = {n: {"model": m, "cv": cv_stub, "train_time": 0}
                         for n, m in models.items()}
    else:
        X_full = np.vstack([X_tr, X_te])
        y_full = np.concatenate([y_tr, y_te])
        train_results = train_all(X_tr, y_tr, X_full, y_full, tune=False)
        models = {n: r["model"] for n, r in train_results.items()}
    return X_tr, X_te, y_tr, y_te, feat_names, scaler, df_raw, models, train_results


# Load dataset and pipelines
X_tr, X_te, y_tr, y_te, feat_names, scaler, df_raw, models, train_results = get_pipeline_and_models()


# ─────────────────────────────────────────────────────────────────────────────
# 3. Dynamic Chart Coloring Helpers for Light/Dark Backdrops
# ─────────────────────────────────────────────────────────────────────────────
def apply_theme_to_fig(fig):
    """
    Strips dark templates and transparentizes background, letting grids/fonts
    adapt automatically to Streamlit's Light or Dark layout.
    """
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(font=dict(color='gray')),
        margin=dict(l=20, r=20, t=30, b=20)
    )
    fig.update_xaxes(gridcolor='rgba(128,128,128,0.15)', zerolinecolor='rgba(128,128,128,0.15)')
    fig.update_yaxes(gridcolor='rgba(128,128,128,0.15)', zerolinecolor='rgba(128,128,128,0.15)')
    return fig


def clip_score(value: float) -> float:
    return float(np.clip(value, 0, 100))


def class_to_score(code: str) -> int:
    return {"L": 35, "M": 65, "H": 90}.get(code, 60)


def get_primary_prediction(results: list) -> dict:
    preferred = next((r for r in results if r["name"] == "Random Forest"), None)
    if preferred:
        return preferred
    scored = [r for r in results if r["confidence"] is not None]
    return max(scored, key=lambda r: r["confidence"]) if scored else results[0]


def predict_primary(raw_input: dict) -> dict:
    return get_primary_prediction(predict_all(raw_input, models))


def build_student_profile(raw_input: dict, results: list) -> dict:
    raised = int(raw_input["raisedhands"])
    visited = int(raw_input["VisITedResources"])
    announce = int(raw_input["AnnouncementsView"])
    discuss = int(raw_input["Discussion"])
    absence_flag = int(str(raw_input["StudentAbsenceDays"]).lower() == "above-7")
    survey_yes = int(str(raw_input["ParentAnsweringSurvey"]).lower() == "yes")
    satisfaction_good = int(str(raw_input["ParentschoolSatisfaction"]).lower() == "good")

    engagement = (raised + visited + announce + discuss) / 4.0
    active_learning = (raised + discuss) / 2.0
    digital_learning = (visited + announce) / 2.0
    communication = (announce + discuss) / 2.0
    parent_support = ((survey_yes + satisfaction_good) / 2.0) * 100.0
    resource_utilization = clip_score((visited * 0.7) + (announce * 0.3))
    consistency = clip_score(100.0 - float(np.std([raised, visited, announce, discuss], ddof=1)))

    success_index = clip_score(
        (engagement * 0.35) +
        (active_learning * 0.20) +
        (resource_utilization * 0.20) +
        (parent_support * 0.15) +
        ((100.0 - absence_flag * 100.0) * 0.10)
    )
    academic_risk = clip_score(
        ((100.0 - engagement) * 0.35) +
        (absence_flag * 100.0 * 0.25) +
        ((100.0 - parent_support) * 0.20) +
        ((100.0 - resource_utilization) * 0.15) +
        ((100.0 - consistency) * 0.05)
    )
    academic_health = clip_score((success_index * 0.6) + ((100.0 - academic_risk) * 0.4))
    primary = get_primary_prediction(results)
    rank_base = {"H": 15, "M": 45, "L": 75}.get(primary["prediction"], 50)
    confidence_bonus = int((primary["confidence"] or 60) / 10)
    estimated_rank = max(5, min(95, rank_base - confidence_bonus))

    if absence_flag:
        persona = "At-Risk Student"
    elif visited >= 75:
        persona = "Resource Explorer"
    elif discuss <= 30 and engagement >= 60:
        persona = "Silent Achiever"
    elif active_learning >= 60 and engagement >= 55:
        persona = "Active Learner"
    else:
        persona = "Balanced Performer"

    badges = []
    if primary["prediction"] == "H":
        badges.append("Class Champion")
    if visited >= 70:
        badges.append("Resource Master")
    if raised >= 70:
        badges.append("Participation Hero")
    if parent_support >= 75:
        badges.append("Parent Supported")
    if consistency >= 75:
        badges.append("Consistent Learner")
    if not badges:
        badges.append("Growth Starter")

    alerts = []
    if absence_flag:
        alerts.append(("High Absence Risk", "Attendance pattern needs attention.", "#ef4444"))
    if parent_support < 50:
        alerts.append(("Low Parent Involvement", "Survey or satisfaction signal is weak.", "#f59e0b"))
    if active_learning < 40:
        alerts.append(("Low Classroom Participation", "Raised hands and discussion are below target.", "#ef4444"))
    if resource_utilization < 40:
        alerts.append(("Low Resource Utilization", "Learning resource activity is below target.", "#f59e0b"))
    if not alerts:
        alerts.append(("Stable Academic Profile", "No major risk alerts triggered.", "#10b981"))

    contributions = [
        ("Resource Utilization", resource_utilization * 0.20, "#10b981"),
        ("Class Participation", active_learning * 0.20, "#2563eb"),
        ("Parent Support", parent_support * 0.15, "#6366f1"),
        ("Attendance", (100 - absence_flag * 100) * 0.10, "#10b981" if not absence_flag else "#ef4444"),
        ("Engagement Balance", consistency * 0.05, "#f59e0b"),
    ]

    recommendations = []
    if discuss < 50:
        recommendations.append("Increase discussion participation to strengthen active learning.")
    if visited < 60:
        recommendations.append("Visit learning resources more frequently during the week.")
    if announce < 50:
        recommendations.append("Check course announcements regularly to avoid missing updates.")
    if absence_flag:
        recommendations.append("Improve attendance; absence is one of the strongest risk signals.")
    if parent_support < 75:
        recommendations.append("Encourage parent survey response and school feedback involvement.")
    if not recommendations:
        recommendations.append("Maintain the current learning pattern and review progress weekly.")

    return {
        "primary": primary,
        "success_index": success_index,
        "academic_risk": academic_risk,
        "academic_health": academic_health,
        "health_label": "Excellent" if academic_health >= 90 else "Good" if academic_health >= 70 else "Average" if academic_health >= 50 else "Needs Attention",
        "persona": persona,
        "badges": badges,
        "alerts": alerts,
        "contributions": contributions,
        "recommendations": recommendations,
        "estimated_rank": estimated_rank,
        "radar": {
            "Participation": raised,
            "Resources": visited,
            "Attendance": 100 - absence_flag * 100,
            "Parent Support": parent_support,
            "Discussion": discuss,
            "Announcements": announce,
        },
        "timeline": {
            "Attendance": 100 - absence_flag * 100,
            "Participation": raised,
            "Resources": visited,
            "Discussion": discuss,
            "Consistency": consistency,
        },
        "what_if": {
            "Raised Hands +30": {**raw_input, "raisedhands": min(100, raised + 30)},
            "Resources +30": {**raw_input, "VisITedResources": min(100, visited + 30)},
            "Discussion +30": {**raw_input, "Discussion": min(100, discuss + 30)},
            "Attendance Improves": {**raw_input, "StudentAbsenceDays": "Under-7"},
            "Parent Support Drops": {**raw_input, "ParentAnsweringSurvey": "No", "ParentschoolSatisfaction": "Bad"},
        },
    }


def build_report_text(raw_input: dict, profile: dict) -> str:
    primary = profile["primary"]
    lines = [
        "Gradelytica Student Report",
        "",
        f"Prediction: {primary['label']} ({primary['prediction']})",
        f"Confidence: {primary['confidence'] if primary['confidence'] is not None else 'N/A'}",
        f"Success Index: {profile['success_index']:.1f}/100",
        f"Academic Risk Index: {profile['academic_risk']:.1f}/100",
        f"Academic Health: {profile['academic_health']:.1f}% ({profile['health_label']})",
        f"Student Persona: {profile['persona']}",
        f"Estimated Rank: Top {profile['estimated_rank']}%",
        "",
        "Input Details:",
    ]
    lines.extend([f"- {k}: {v}" for k, v in raw_input.items()])
    lines.append("")
    lines.append("Recommendations:")
    lines.extend([f"- {rec}" for rec in profile["recommendations"]])
    return "\n".join(lines)


NATIONALITY_ISO3 = {
    "Egypt": "EGY", "Iran": "IRN", "Iraq": "IRQ", "Jordan": "JOR",
    "KW": "KWT", "Kuwait": "KWT", "Lebanon": "LBN", "lebanon": "LBN",
    "Lybia": "LBY", "Morocco": "MAR", "Palestine": "PSE",
    "SaudiArabia": "SAU", "Syria": "SYR", "Tunis": "TUN",
    "USA": "USA", "venzuela": "VEN",
}


# ─────────────────────────────────────────────────────────────────────────────
# 4. Premium Sidebar Navigation (Adaptive and Elegant)
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.markdown("<h2 style='margin-bottom:0; color: #2563eb; font-weight: 700; letter-spacing: -0.5px;'>Gradelytica</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='opacity:0.7; font-size:0.82rem; margin-top:2px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;'>Predictive Analytics Engine</p>", unsafe_allow_html=True)

page = st.sidebar.radio(
    "NAVIGATION",
    ["Overview", "Model Performance", "Interactive Predictor", "Data Visualizations"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("### SYSTEM DETAILS")
st.sidebar.text(f"Engineered Features: {len(feat_names)}")
st.sidebar.text(f"Cross-Validation: {cfg.CV_FOLDS}-Fold Stratified")
st.sidebar.text(f"Models: {len(models)} Operational")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1: Overview
# ═══════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.title("Data Overview & Registry")
    st.markdown("Explore core metrics, browse filtered slices of the registry, and view feature correlations.")

    # Responsive stat cards wrapping cleanly
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    with c1:
        st.markdown(f'<div class="stat-card"><h3>{df_raw.shape[0]}</h3><p>Total Records</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="stat-card"><h3>{df_raw.shape[1]}</h3><p>Input Variables</p></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="stat-card"><h3>{df_raw.isnull().sum().sum()}</h3><p>Missing Entries</p></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="stat-card"><h3>{df_raw["Class"].nunique()}</h3><p>Output Classes</p></div>', unsafe_allow_html=True)

    st.markdown("<h3 style='margin-top:35px;'>Interactive Cohort Filtering</h3>", unsafe_allow_html=True)
    col_a, col_b, col_c = st.columns(3)
    
    gender_f = col_a.multiselect("Gender", df_raw["gender"].unique(), default=list(df_raw["gender"].unique()))
    nat_f    = col_b.multiselect("Student Nationality", sorted(df_raw["NationalITy"].unique()),
                                  default=list(df_raw["NationalITy"].unique()))
    topic_f  = col_c.multiselect("Topic Area", sorted(df_raw["Topic"].unique()),
                                  default=list(df_raw["Topic"].unique()))
    
    filtered = df_raw[(df_raw["gender"].isin(gender_f)) &
                      (df_raw["NationalITy"].isin(nat_f)) &
                      (df_raw["Topic"].isin(topic_f))]
    
    st.dataframe(filtered, use_container_width=True, height=280)

    st.markdown("### Numerical Feature Correlation Matrix")
    corr = df_raw.corr(numeric_only=True)
    
    fig_h = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        aspect="auto"
    )
    apply_theme_to_fig(fig_h)
    fig_h.update_layout(height=400)
    st.plotly_chart(fig_h, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2: Model Performance
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Model Performance":
    st.title("Model Performance & Diagnostics")
    st.markdown("Inspect performance indexes, cross-validation boundaries, and structural metrics across all active models.")

    # Calculate metrics
    all_metrics = []
    for name, r in train_results.items():
        m = compute_metrics(name, r["model"], X_te, y_te, r["cv"])
        all_metrics.append(m)

    tab_table, tab_charts, tab_matrix, tab_roc = st.tabs([
        "Registry & Metrics", "Accuracy Comparison", "Confusion Matrices", "AUC-ROC Diagnostics"
    ])

    with tab_table:
        rows = []
        for m in all_metrics:
            rows.append({
                "Model Algorithm": m["name"],
                "Test Accuracy": f"{m['accuracy'] * 100:.2f}%",
                "Weighted Precision": f"{m['precision'] * 100:.2f}%",
                "Weighted Recall": f"{m['recall'] * 100:.2f}%",
                "Weighted F1": f"{m['f1'] * 100:.2f}%",
                "Macro AUC-ROC": f"{m['auc_roc'] * 100:.2f}%" if m["auc_roc"] else "N/A",
                "Cross-Val Accuracy": m["cv_accuracy"],
            })
        st.dataframe(pd.DataFrame(rows).set_index("Model Algorithm"), use_container_width=True)

    with tab_charts:
        names = [m["name"] for m in all_metrics]
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            name="Test Accuracy",
            x=names,
            y=[m["accuracy"] for m in all_metrics],
            marker_color="#2563eb"
        ))
        fig_bar.add_trace(go.Bar(
            name="F1-Score",
            x=names,
            y=[m["f1"] for m in all_metrics],
            marker_color="#10b981"
        ))
        fig_bar.update_layout(barmode="group", yaxis_range=[0, 1.05], height=400)
        apply_theme_to_fig(fig_bar)
        st.plotly_chart(fig_bar, use_container_width=True)

    with tab_matrix:
        names = [m["name"] for m in all_metrics]
        sel = st.selectbox("Select Model Algorithm", names, key="cm_sel")
        sel_m = next(m for m in all_metrics if m["name"] == sel)
        from sklearn.metrics import confusion_matrix as sk_cm
        cm = sk_cm(y_te, sel_m["y_pred"])
        
        fig_cm = px.imshow(
            cm,
            text_auto=True,
            x=["Predicted Low", "Predicted Medium", "Predicted High"],
            y=["Actual Low", "Actual Medium", "Actual High"],
            color_continuous_scale="Blues",
        )
        apply_theme_to_fig(fig_cm)
        fig_cm.update_layout(height=400)
        st.plotly_chart(fig_cm, use_container_width=True)

    with tab_roc:
        names = [m["name"] for m in all_metrics]
        from sklearn.metrics import auc, roc_curve as sk_roc
        cls_pick = st.radio("Target Prediction State Class", ["Low (L)", "Medium (M)", "High (H)"], horizontal=True)
        cls_idx = {"Low (L)": 0, "Medium (M)": 1, "High (H)": 2}[cls_pick]
        y_bin = (y_te == cls_idx).astype(int)

        fig_roc = go.Figure()
        for m in all_metrics:
            if m["y_proba"] is not None:
                fpr, tpr, _ = sk_roc(y_bin, m["y_proba"][:, cls_idx])
                auc_val = auc(fpr, tpr)
                fig_roc.add_trace(go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode="lines",
                    name=f"{m['name']} (AUC = {auc_val:.3f})"
                ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode="lines",
            line=dict(dash="dash", color="#94a3b8"),
            name="Random Guess Bound"
        ))
        fig_roc.update_layout(
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=420
        )
        apply_theme_to_fig(fig_roc)
        st.plotly_chart(fig_roc, use_container_width=True)

    st.markdown("<h3 style='margin-top:35px;'>Feature Importance Profile</h3>", unsafe_allow_html=True)
    tree_names = [n for n in names if n in ("Decision Tree", "Random Forest", "Gradient Boosting")]
    if tree_names:
        t_sel = st.selectbox("Select Model Structure", tree_names, key="fi_sel")
        mdl = models[t_sel]
        clf = mdl.named_steps["clf"] if hasattr(mdl, "named_steps") else mdl
        if hasattr(clf, "feature_importances_"):
            imp = clf.feature_importances_
            fi_df = pd.DataFrame({"Feature Name": feat_names, "Relative Weight": imp}).sort_values("Relative Weight")
            
            fig_fi = px.bar(
                fi_df,
                x="Relative Weight",
                y="Feature Name",
                orientation="h",
                color="Relative Weight",
                color_continuous_scale="Blues"
            )
            apply_theme_to_fig(fig_fi)
            fig_fi.update_layout(height=450)
            st.plotly_chart(fig_fi, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3: Interactive Predictor
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Interactive Predictor":
    st.title("Interactive Student Predictor")
    st.markdown("Supply features corresponding to student profile and track real-time inferences.")

    with st.form("predict_form"):
        st.markdown("#### Demographics & Environment")
        c1, c2, c3 = st.columns(3)
        gender = c1.selectbox("Student Gender", ["M", "F"])
        nat    = c2.selectbox("Student Nationality", sorted(df_raw["NationalITy"].unique()))
        pob    = c3.selectbox("Place of Birth", sorted(df_raw["PlaceofBirth"].unique()))

        c4, c5, c6 = st.columns(3)
        stage    = c4.selectbox("Educational Stage", ["lowerlevel", "MiddleSchool", "HighSchool"])
        grade    = c5.selectbox("Grade Index", [f"G-{i:02d}" for i in range(2, 13)])
        section  = c6.selectbox("Assigned Section", sorted(df_raw["SectionID"].unique()))

        c7, c8, c9 = st.columns(3)
        topic    = c7.selectbox("Topic Subject", sorted(df_raw["Topic"].unique()))
        semester = c8.selectbox("Current Semester", ["F", "S"])
        relation = c9.selectbox("Responsible Parent Relation", ["Father", "Mum"])

        st.markdown("<hr style='margin: 20px 0; border:0; border-top: 1px solid rgba(128,128,128,0.15);'>", unsafe_allow_html=True)
        st.markdown("#### Classroom Engagement Metrics")
        e1, e2, e3, e4 = st.columns(4)
        raised   = e1.slider("Raised Hands Index", 0, 100, 40)
        visited  = e2.slider("Resource Visits Index", 0, 100, 45)
        announce = e3.slider("Announcements Consulted", 0, 100, 30)
        discuss  = e4.slider("Active Discussions", 0, 100, 35)

        st.markdown("<hr style='margin: 20px 0; border:0; border-top: 1px solid rgba(128,128,128,0.15);'>", unsafe_allow_html=True)
        st.markdown("#### Family Survey & Attendance")
        p1, p2, p3 = st.columns(3)
        survey  = p1.selectbox("Parent Responded to Survey", ["Yes", "No"])
        sat     = p2.selectbox("Parent School Satisfaction Rating", ["Good", "Bad"])
        absence = p3.selectbox("Absenteeism Flags", ["Under-7", "Above-7"])

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("Generate Predictive Report", use_container_width=True)

    if submitted:
        st.session_state["last_student_input"] = {
            "gender": gender, "NationalITy": nat, "PlaceofBirth": pob,
            "StageID": stage, "GradeID": grade, "SectionID": section,
            "Topic": topic, "Semester": semester, "Relation": relation,
            "raisedhands": raised, "VisITedResources": visited,
            "AnnouncementsView": announce, "Discussion": discuss,
            "ParentAnsweringSurvey": survey,
            "ParentschoolSatisfaction": sat,
            "StudentAbsenceDays": absence,
        }

    if "last_student_input" in st.session_state:
        raw_input = dict(st.session_state["last_student_input"])
        raised = int(raw_input["raisedhands"])
        visited = int(raw_input["VisITedResources"])
        discuss = int(raw_input["Discussion"])
        absence = raw_input["StudentAbsenceDays"]

        results = predict_all(raw_input, models)
        profile = build_student_profile(raw_input, results)
        primary = profile["primary"]

        st.markdown("<h3 style='margin-top:35px;'>Student Success Center</h3>", unsafe_allow_html=True)
        meter_col, health_col, persona_col = st.columns([1.25, 1, 1])
        with meter_col:
            fig_meter = go.Figure(go.Indicator(
                mode="gauge+number",
                value=profile["success_index"],
                number={"suffix": "/100"},
                title={"text": f"Success Meter - {primary['label']}"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#2563eb"},
                    "steps": [
                        {"range": [0, 50], "color": "rgba(239, 68, 68, 0.20)"},
                        {"range": [50, 70], "color": "rgba(245, 158, 11, 0.20)"},
                        {"range": [70, 100], "color": "rgba(16, 185, 129, 0.20)"},
                    ],
                },
            ))
            apply_theme_to_fig(fig_meter)
            fig_meter.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=10))
            st.plotly_chart(fig_meter, use_container_width=True)
        with health_col:
            health_color = "#10b981" if profile["academic_health"] >= 70 else "#f59e0b" if profile["academic_health"] >= 50 else "#ef4444"
            st.markdown(f"""
            <div class="pred-card">
                <div style="font-size:0.78rem; opacity:0.72; font-weight:700; text-transform:uppercase;">Academic Health</div>
                <div style="font-size:2.1rem; font-weight:700; color:{health_color};">{profile['academic_health']:.1f}%</div>
                <div style="font-weight:600;">{profile['health_label']}</div>
                <div class="progress-container"><div class="progress-bar-fill" style="width:{profile['academic_health']}%; background-color:{health_color};"></div></div>
                <div style="margin-top:12px; font-size:0.86rem;">Estimated Rank: <b>Top {profile['estimated_rank']}%</b></div>
            </div>
            """, unsafe_allow_html=True)
        with persona_col:
            confidence = primary["confidence"] if primary["confidence"] is not None else 0
            conf_color = "#10b981" if confidence >= 80 else "#f59e0b" if confidence >= 60 else "#ef4444"
            st.markdown(f"""
            <div class="pred-card">
                <div style="font-size:0.78rem; opacity:0.72; font-weight:700; text-transform:uppercase;">Student Persona</div>
                <div style="font-size:1.45rem; font-weight:700; color:#2563eb;">{profile['persona']}</div>
                <div style="margin-top:10px; font-size:0.86rem;">Primary model: <b>{primary['name']}</b></div>
                <div style="margin-top:6px; font-size:0.86rem;">Confidence: <b style="color:{conf_color};">{primary['confidence'] if primary['confidence'] is not None else 'N/A'}%</b></div>
            </div>
            """, unsafe_allow_html=True)

        alert_cols = st.columns(3)
        for i, (title, detail, color) in enumerate(profile["alerts"][:3]):
            with alert_cols[i % 3]:
                st.markdown(f"""
                <div class="pred-card" style="border-left:4px solid {color};">
                    <div style="font-weight:700; color:{color};">{title}</div>
                    <div style="font-size:0.86rem; opacity:0.78; margin-top:4px;">{detail}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<h3 style='margin-top:24px;'>Achievement Badges</h3>", unsafe_allow_html=True)
        st.markdown(" ".join([
            f'<span class="badge badge-high" style="margin:4px 6px 4px 0;">{badge}</span>'
            for badge in profile["badges"]
        ]), unsafe_allow_html=True)

        st.markdown("<h3 style='margin-top:35px;'>Operational Prediction Outputs</h3>", unsafe_allow_html=True)
        
        # Responsive grid structures for prediction cards (wraps on mobile screen widths)
        card_cols = st.columns(2)
        for i, r in enumerate(results):
            col_target = card_cols[0] if i % 2 == 0 else card_cols[1]
            
            badge_class = "badge-medium"
            if r["prediction"] == "H":
                badge_class = "badge-high"
            elif r["prediction"] == "L":
                badge_class = "badge-low"
                
            color_fill = "#ef4444"
            if r["prediction"] == "H":
                color_fill = "#10b981"
            elif r["prediction"] == "M":
                color_fill = "#f59e0b"
                
            conf_percent = r["confidence"] if r["confidence"] is not None else 0
            conf_str = f"Confidence: {conf_percent}%" if r["confidence"] is not None else "Probability: N/A"
            
            with col_target:
                st.markdown(f"""
                <div class="pred-card">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span style="font-weight:600; font-size:1.02rem; color:var(--text-color);">{r['name']}</span>
                        <span class="badge {badge_class}">{r['label']}</span>
                    </div>
                    <div style="display:flex; justify-content:space-between; margin-top:12px; font-size:0.78rem; opacity:0.75;">
                        <span>Predictive Confidence</span>
                        <span>{conf_str}</span>
                    </div>
                    <div class="progress-container">
                        <div class="progress-bar-fill" style="width: {conf_percent}%; background-color: {color_fill};"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        conf_data = [r for r in results if r["confidence"] is not None]
        if conf_data:
            st.markdown("<h3 style='margin-top:30px;'>Confidence Matrix</h3>", unsafe_allow_html=True)
            
            fig_c = px.bar(
                x=[r["name"] for r in conf_data],
                y=[r["confidence"] for r in conf_data],
                color=[r["prediction"] for r in conf_data],
                color_discrete_map={"H": "#10b981", "M": "#f59e0b", "L": "#ef4444"},
                labels={"x": "Active Model", "y": "Calculated Confidence %"},
            )
            apply_theme_to_fig(fig_c)
            fig_c.update_layout(yaxis_range=[0, 105], height=340)
            st.plotly_chart(fig_c, use_container_width=True)

        tab_explain, tab_sim, tab_radar, tab_report = st.tabs([
            "Advisor & Contributions", "What-If Simulator", "Radar & Timeline", "Report Download"
        ])

        with tab_explain:
            c_left, c_right = st.columns([1, 1])
            with c_left:
                st.markdown("#### Feature Contribution Breakdown")
                contrib_df = pd.DataFrame(profile["contributions"], columns=["Factor", "Impact", "Color"])
                fig_contrib = go.Figure(go.Bar(
                    x=contrib_df["Impact"],
                    y=contrib_df["Factor"],
                    orientation="h",
                    marker_color=contrib_df["Color"],
                    text=[f"+{v:.1f}" for v in contrib_df["Impact"]],
                    textposition="auto",
                ))
                fig_contrib.update_layout(height=300, xaxis_title="Positive score contribution")
                apply_theme_to_fig(fig_contrib)
                st.plotly_chart(fig_contrib, use_container_width=True)
            with c_right:
                st.markdown("#### AI Academic Advisor")
                for rec in profile["recommendations"]:
                    st.markdown(f"- {rec}")

                st.markdown("#### Academic Risk Index")
                risk_color = "#ef4444" if profile["academic_risk"] >= 60 else "#f59e0b" if profile["academic_risk"] >= 35 else "#10b981"
                st.markdown(f"""
                <div class="pred-card">
                    <div style="font-size:1.8rem; font-weight:700; color:{risk_color};">{profile['academic_risk']:.1f}/100</div>
                    <div class="progress-container"><div class="progress-bar-fill" style="width:{profile['academic_risk']}%; background-color:{risk_color};"></div></div>
                </div>
                """, unsafe_allow_html=True)

        with tab_sim:
            st.markdown("#### Academic Twin Simulator")
            twin_rows = []
            current = primary
            for scenario, twin_input in profile["what_if"].items():
                twin_prediction = predict_primary(twin_input)
                twin_rows.append({
                    "Scenario": scenario,
                    "Prediction": f"{current['label']} -> {twin_prediction['label']}",
                    "Confidence": f"{twin_prediction['confidence']}%" if twin_prediction["confidence"] is not None else "N/A",
                    "Model": twin_prediction["name"],
                })
            st.dataframe(pd.DataFrame(twin_rows), use_container_width=True)

            st.markdown("#### Manual What-If Controls")
            sim_raised = st.slider("Simulated Raised Hands", 0, 100, raised, key="sim_raised")
            sim_resources = st.slider("Simulated Resource Visits", 0, 100, visited, key="sim_resources")
            sim_discussion = st.slider("Simulated Discussion", 0, 100, discuss, key="sim_discussion")
            sim_absence = st.selectbox("Simulated Absence", ["Under-7", "Above-7"], index=0 if absence == "Under-7" else 1, key="sim_absence")
            sim_input = {
                **raw_input,
                "raisedhands": sim_raised,
                "VisITedResources": sim_resources,
                "Discussion": sim_discussion,
                "StudentAbsenceDays": sim_absence,
            }
            sim_prediction = predict_primary(sim_input)
            sim_conf = sim_prediction["confidence"] if sim_prediction["confidence"] is not None else "N/A"
            current_conf = primary["confidence"] if primary["confidence"] is not None else "N/A"
            st.markdown(f"""
            <div class="pred-card">
                <div style="font-size:0.82rem; opacity:0.72; font-weight:700; text-transform:uppercase;">What-If Result</div>
                <div style="font-size:1.3rem; font-weight:700;">{primary['label']} -> {sim_prediction['label']}</div>
                <div style="font-size:0.9rem; opacity:0.78;">Confidence: {current_conf}% -> {sim_conf}%</div>
            </div>
            """, unsafe_allow_html=True)

        with tab_radar:
            radar_labels = list(profile["radar"].keys())
            radar_values = list(profile["radar"].values())
            fig_radar = go.Figure(go.Scatterpolar(
                r=radar_values + [radar_values[0]],
                theta=radar_labels + [radar_labels[0]],
                fill="toself",
                name="Academic profile",
                line_color="#2563eb",
            ))
            fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), height=420)
            apply_theme_to_fig(fig_radar)
            st.plotly_chart(fig_radar, use_container_width=True)

            st.markdown("#### Achievement Timeline")
            for label, value in profile["timeline"].items():
                bar_color = "#10b981" if value >= 70 else "#f59e0b" if value >= 45 else "#ef4444"
                st.markdown(f"""
                <div style="margin-bottom:10px;">
                    <div style="display:flex; justify-content:space-between; font-size:0.88rem;">
                        <span>{label}</span><span>{value:.0f}%</span>
                    </div>
                    <div class="progress-container"><div class="progress-bar-fill" style="width:{value}%; background-color:{bar_color};"></div></div>
                </div>
                """, unsafe_allow_html=True)

        with tab_report:
            report_text = build_report_text(raw_input, profile)
            st.text_area("Student Report Preview", report_text, height=360)
            st.download_button(
                "Download Student Report",
                data=report_text,
                file_name="gradelytica_student_report.txt",
                mime="text/plain",
                use_container_width=True,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4: Data Visualizations
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Data Visualizations":
    st.title("Exploratory Data Visualizations")
    st.markdown("Review raw features distributions, cross-categoricals, and pair relations interactively.")
    
    COLOR_MAP = {"L": "#ef4444", "M": "#f59e0b", "H": "#10b981"}

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Distribution Slices", "Cross-Category Analysis", "Engagement Relatables", "Multi-Variable Matrix", "Nationality Map"
    ])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            fig1 = px.histogram(
                df_raw, x="Class", color="Class",
                category_orders={"Class": ["L", "M", "H"]},
                color_discrete_map=COLOR_MAP,
                title="Performance Level Distribution"
            )
            apply_theme_to_fig(fig1)
            fig1.update_layout(height=380)
            st.plotly_chart(fig1, use_container_width=True)
            
        with c2:
            fig4 = px.pie(
                df_raw, names="Class", color="Class",
                color_discrete_map=COLOR_MAP,
                title="Percentage Split", hole=0.55
            )
            apply_theme_to_fig(fig4)
            fig4.update_layout(height=380)
            st.plotly_chart(fig4, use_container_width=True)

    with tab2:
        cat = st.selectbox("Categorical Attribute to Segment By", [
            "gender", "Semester", "NationalITy", "StageID",
            "Topic", "SectionID", "StudentAbsenceDays", "GradeID", "Relation"
        ])
        fig2 = px.histogram(
            df_raw, x=cat, color="Class",
            category_orders={"Class": ["L", "M", "H"]},
            color_discrete_map=COLOR_MAP, barmode="group",
            title=f"Class Segmentations across {cat}"
        )
        apply_theme_to_fig(fig2)
        fig2.update_layout(height=420)
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        metric = st.selectbox("Select Numerical Target", ["raisedhands", "VisITedResources", "AnnouncementsView", "Discussion"])
        fig3 = px.box(
            df_raw, x="Class", y=metric, color="Class",
            category_orders={"Class": ["L", "M", "H"]},
            color_discrete_map=COLOR_MAP,
            title=f"{metric} Spread per Target State"
        )
        apply_theme_to_fig(fig3)
        fig3.update_layout(height=380)
        st.plotly_chart(fig3, use_container_width=True)

    with tab4:
        st.markdown("#### Numeric Pair Relationships Scatter Matrix")
        fig5 = px.scatter_matrix(
            df_raw,
            dimensions=["raisedhands", "VisITedResources", "AnnouncementsView", "Discussion"],
            color="Class",
            color_discrete_map=COLOR_MAP,
        )
        apply_theme_to_fig(fig5)
        fig5.update_layout(height=600)
        fig5.update_traces(diagonal_visible=False, marker=dict(size=4.5, opacity=0.55))
        st.plotly_chart(fig5, use_container_width=True)

    with tab5:
        st.markdown("#### Average Performance by Student Nationality")
        map_df = df_raw.copy()
        map_df["iso_alpha"] = map_df["NationalITy"].map(NATIONALITY_ISO3)
        map_df["performance_score"] = map_df["Class"].map(class_to_score)
        map_df = (
            map_df.dropna(subset=["iso_alpha"])
            .groupby(["NationalITy", "iso_alpha"], as_index=False)
            .agg(
                average_performance=("performance_score", "mean"),
                students=("Class", "count"),
            )
        )
        fig_map = px.choropleth(
            map_df,
            locations="iso_alpha",
            color="average_performance",
            hover_name="NationalITy",
            hover_data={"students": True, "average_performance": ":.1f", "iso_alpha": False},
            color_continuous_scale=["#ef4444", "#f59e0b", "#10b981"],
            range_color=[30, 90],
            title="Average Performance by Country",
        )
        apply_theme_to_fig(fig_map)
        fig_map.update_layout(height=500, geo=dict(showframe=False, showcoastlines=True))
        st.plotly_chart(fig_map, use_container_width=True)
        st.dataframe(map_df.sort_values("average_performance", ascending=False), use_container_width=True)
