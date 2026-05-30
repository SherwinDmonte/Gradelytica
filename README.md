# 🎓 Gradelytica - Student Performance Prediction 

A professional machine-learning pipeline that predicts student academic performance (Low / Medium / High) using **7 classifiers**, **35 total model features**, and a **Streamlit interactive dashboard**.

---

## 🚀 Features

| Category | Details |
|---|---|
| **Models (7)** | Decision Tree · Random Forest · Gradient Boosting · SVM · Logistic Regression · Perceptron · MLP Neural Network |
| **Preprocessing** | All 16 original features used + 19 engineered features for engagement, activity balance, parent support, attendance risk, and success/risk indexes |
| **Evaluation** | Accuracy, Precision, Recall, F1, AUC-ROC, 5-fold stratified cross-validation |
| **Tuning** | RandomizedSearchCV for Random Forest & Gradient Boosting |
| **Explainability** | Feature importance charts, SHAP summary plots |
| **Dashboard** | 4-page Streamlit app with prediction, diagnostics, what-if simulation, and interactive Plotly charts |
| **Persistence** | Trained models saved/loaded via joblib |
| **CLI** | Full command-line interface with menus preserved |

---

## 📁 Project Structure

```
StudentPerformancePrediction-ML/
├── data/
│   └── AI-Data.csv            # Dataset (490 students, 17 columns)
├── models/                    # Saved trained models (.pkl)
├── outputs/                   # Generated charts and reports (.png)
├── src/
│   ├── __init__.py
│   ├── preprocessing.py       # Data loading, feature engineering, encoding, splitting
│   ├── train.py               # Model training, cross-val, hyperparameter tuning
│   ├── evaluate.py            # Metrics, confusion matrices, ROC, learning curves
│   ├── predict.py             # Single-student prediction logic
│   ├── visualize.py           # Static EDA chart generation (matplotlib/seaborn)
│   └── explainability.py      # Feature importance & SHAP
├── app.py                     # Streamlit web dashboard (main entry)
├── cli.py                     # Command-line interface
├── config.py                  # Centralised settings & hyperparameters
├── Project.py                 # Original legacy script (kept for reference)
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

**AI-Data.csv** — 490 student records with 16 features + 1 target (`Class`).

The current dataset contains 3 missing values, so the preprocessing pipeline drops those rows and trains/evaluates on 487 complete records.

| Column | Type | Description |
|--------|------|-------------|
| gender | Categorical | M / F |
| NationalITy | Categorical | Student nationality |
| PlaceofBirth | Categorical | Country of birth |
| StageID | Categorical | lowerlevel / MiddleSchool / HighSchool |
| GradeID | Categorical | G-02 through G-12 |
| SectionID | Categorical | A / B / C |
| Topic | Categorical | Course subject (IT, Math, Science, …) |
| Semester | Categorical | F (First) / S (Second) |
| Relation | Categorical | Responsible parent (Father / Mum) |
| raisedhands | Numeric 0-100 | Times student raised hand in class |
| VisITedResources | Numeric 0-100 | Times student visited course content |
| AnnouncementsView | Numeric 0-100 | Times student viewed announcements |
| Discussion | Numeric 0-100 | Times student participated in discussions |
| ParentAnsweringSurvey | Categorical | Yes / No |
| ParentschoolSatisfaction | Categorical | Good / Bad |
| StudentAbsenceDays | Categorical | Under-7 / Above-7 |
| **Class** | **Target** | **L (Low) / M (Medium) / H (High)** |

---

## 🧩 Engineered Features

The training pipeline adds 19 derived model features:

| Feature | Meaning |
|---------|---------|
| engagement_score | Average of raised hands, resource visits, announcements viewed, and discussion |
| parent_involvement | Combined parent survey and satisfaction signal |
| absence_flag | 1 when absences are Above-7, else 0 |
| activity_total | Sum of the four activity metrics |
| active_learning_score | Average of raised hands and discussion |
| digital_learning_score | Average of visited resources and announcements viewed |
| communication_score | Average of announcements viewed and discussion |
| engagement_consistency | Stability of activity metrics across participation areas |
| support_risk_score | Combined absence and low parent-support risk |
| engagement_trend_category | Encoded Low / Medium / High engagement band |
| student_success_index | Weighted success score from engagement, resources, support, and attendance |
| academic_risk_index | Weighted risk score from low engagement, absence, low support, and low resources |
| parent_support_score | Parent support score scaled from 0 to 100 |
| activity_consistency_score | Consistency score scaled from 0 to 100 |
| active_learner_flag | 1 when active learning and engagement are strong |
| passive_learner_flag | 1 when active learning and engagement are low |
| digital_engagement_ratio | Digital activity share of total activity |
| seniority_score | Grade level scaled from 0 to 100 |
| resource_utilization_score | Weighted score for resource visits and announcement views |

---

## 🖥️ Streamlit Dashboard Highlights

The dashboard includes:

| Feature | Description |
|---------|-------------|
| Student Success Meter | Gauge chart showing a custom 0-100 success score |
| Academic Health Score | Progress score with Excellent / Good / Average / Needs Attention bands |
| Academic Risk Alerts | Color-coded warnings for absence, low participation, low resources, and parent support |
| Student Persona Generator | Classifies students as Active Learner, Resource Explorer, Silent Achiever, At-Risk Student, or Balanced Performer |
| Gamification Badges | Awards badges such as Class Champion, Resource Master, Participation Hero, and Consistent Learner |
| Feature Contribution Breakdown | Credit-score-style explanation of positive academic factors |
| What-If Simulator | Lets users adjust key behaviors and compare current vs simulated predictions |
| Academic Twin Simulator | Shows scenario-based prediction changes for attendance, resources, discussion, and parent support |
| AI Academic Advisor | Generates rule-based recommendations from weak student signals |
| Performance Radar Chart | Visualizes participation, resources, attendance, parent support, discussion, and announcements |
| Nationality Performance Map | Choropleth map showing average performance by student nationality |
| Student Report Download | Downloads a text-based student report with prediction, scores, inputs, and recommendations |

---

## 🧠 Model Results

Latest saved-model performance on the current processed dataset:

| Model | Accuracy | F1-Score |
|-------|----------|----------|
| Decision Tree | 0.6667 | 0.6649 |
| Random Forest | 0.8027 | 0.8030 |
| Gradient Boosting | 0.7755 | 0.7758 |
| SVM | 0.7551 | 0.7549 |
| Logistic Regression | 0.7075 | 0.7064 |
| Perceptron | 0.5510 | 0.5328 |
| MLP Classifier | 0.7415 | 0.7414 |

---
