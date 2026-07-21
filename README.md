# 🩺 Multi-Disease Risk Prediction System
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red)
![ML](https://img.shields.io/badge/ML-Ensemble%20(RF%2FKNN%2FSVM%2FNB)-green)

## 🔗 Live Demo
👉 **[Click here to open the app](https://disease-prediction-app-roavq7vmbzjgne8xhpkoml.streamlit.app/)**

---

## 📌 Overview
An end-to-end two-stage machine learning system that predicts risk
for Heart Disease, Diabetes, and Chronic Kidney Disease, combining
a high-precision filter with a heterogeneous ensemble, plus an
explanation layer that tells the user *why* a prediction was made.

---

## 🎯 Features
- **Disease Selector** — pick from Heart Disease, Diabetes, or CKD via clickable cards
- **Dynamic Input Form** — only the fields each disease's model actually needs
- **Two-Stage Prediction Engine** — fast Random Forest filter + 3-model ensemble for uncertain cases
- **Confidence Score** — probability-based output, not just a binary label
- **Explanation Layer** — top contributing factors + plain-language clinical advice for positive results

---

## 🤖 ML Model

| Detail | Value |
|--------|-------|
| Architecture | Two-stage: Random Forest (Stage 1) → KNN + SVM + Naive Bayes ensemble (Stage 2) |
| Combination Rule | Majority vote (≥2 of 3 models agree) |
| Preprocessing | StandardScaler (KNN/SVM only — RF/NB use raw features) |
| Threshold Tuning | Per-model, per-disease, via precision-recall curve analysis |
| Diseases | Heart Disease / Diabetes / Chronic Kidney Disease |

| Disease | Precision | Recall |
|---|---|---|
| Heart Disease | 0.90 | 0.88 |
| Diabetes | 0.72 | 0.85 |
| Chronic Kidney Disease | 1.00 | 1.00 |

> **Context:** Diabetes prediction excludes HbA1c (the direct
> diagnostic marker) to avoid label leakage, making it the hardest
> of the three tasks — the model relies on indirect risk factors only.
> CKD's near-perfect score reflects the dataset's high separability
> given strong direct clinical markers, consistent with published
> benchmarks on this dataset.

> **Fix applied:** The initial ensemble used an OR-rule (any one
> model disagreeing flipped the result to "disease"), which let
> false positives compound across all three models — diabetes
> precision dropped to 0.32. Switching to a majority-vote rule
> was tested empirically across all three diseases and improved
> precision substantially (e.g., heart disease: 0.70 → 0.90) for
> only marginal recall cost.

---

## 🛠️ Tech Stack
- **Language:** Python
- **Dashboard:** Streamlit
- **ML:** Scikit-learn
- **Data:** Pandas, NumPy
- **Model Persistence:** joblib

---

## 📁 Datasets
- Heart Disease — UCI Heart Disease (Cleveland) dataset
- Diabetes — Vanderbilt Diabetes dataset
- Chronic Kidney Disease — UCI Risk Factor Prediction of CKD dataset

[🔗 Heart Disease Dataset](https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci)
[🔗 Diabetes Dataset](https://www.kaggle.com/datasets/imtkaggleteam/diabetes)
[🔗 CKD Dataset](https://www.kaggle.com/datasets/jhs070701/uci-new-chronic-kidney-dataset-aug-2023-released)

---

## 🚀 Run Locally
git clone: https://github.com/thedivineson/disease-prediction-app
cd disease-risk-predictor
pip install -r requirements.txt
streamlit run app.py

## 📷 Screenshots


