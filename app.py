"""
Multi-Disease Prediction App
Diseases: Heart Disease, Diabetes, Chronic Kidney Disease (CKD)

Expects the following files to sit in the SAME folder as this script
(these are produced by the training notebook via joblib.dump / json.dump):

  heart_rf.pkl      heart_knn.pkl      heart_svm.pkl      heart_nb.pkl      heart_scaler.pkl      heart_thresholds.json
  diabetes_rf.pkl   diabetes_knn.pkl   diabetes_svm.pkl   diabetes_nb.pkl   diabetes_scaler.pkl   diabetes_thresholds.json
  ckd_rf.pkl        ckd_knn.pkl        ckd_svm.pkl        ckd_nb.pkl        ckd_scaler.pkl        ckd_thresholds.json
"""

import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------------------------------------
st.set_page_config(page_title="Disease Risk Predictor", page_icon="🩺", layout="wide")

# --------------------------------------------------------------------------------------
# STYLE  (white background, royal blue accent, dark grey text, light grey secondary)
# --------------------------------------------------------------------------------------
ROYAL_BLUE = "#2A4494"       # slightly dark royal blue
ROYAL_BLUE_HOVER = "#213570"
DARK_GREY = "#333333"
LIGHT_GREY = "#8A8A8A"
BORDER_GREY = "#D9D9D9"

st.markdown(
    f"""
    <style>
        /* ---- Overall canvas ---- */
        .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{
            background-color: #FFFFFF !important;
        }}
        .block-container {{
            max-width: 1000px;
            padding-top: 2.5rem;
            padding-bottom: 3rem;
        }}

        html, body, [class*="css"] {{
            color: {DARK_GREY} !important;
            font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
        }}

        /* ---- Title ---- */
        .app-title {{
            color: {ROYAL_BLUE};
            font-size: 3rem;
            font-weight: 800;
            line-height: 1.15;
            margin-bottom: 0.3rem;
        }}
        .app-subtitle {{
            color: {LIGHT_GREY};
            font-size: 1.05rem;
            margin-bottom: 2.4rem;
        }}

        /* ---- Section headers ---- */
        .section-header {{
            color: {ROYAL_BLUE};
            font-size: 1.55rem;
            font-weight: 700;
            margin-top: 1.8rem;
            margin-bottom: 1.1rem;
            padding-bottom: 0.6rem;
            border-bottom: 3px solid {ROYAL_BLUE};
        }}

        /* ---- Field labels ---- */
        label, .stSelectbox label, .stNumberInput label, .stRadio label {{
            color: {DARK_GREY} !important;
            font-weight: 600 !important;
            font-size: 1rem !important;
        }}

        /* ---- Number inputs: force white bg / dark grey text ---- */
        .stNumberInput input,
        [data-testid="stNumberInput"] input {{
            background-color: #FFFFFF !important;
            color: {DARK_GREY} !important;
            border: 1px solid {BORDER_GREY} !important;
            border-radius: 8px !important;
            font-size: 1.05rem !important;
            padding: 0.55rem 0.7rem !important;
        }}
        .stNumberInput button, [data-testid="stNumberInput"] button {{
            background-color: #FFFFFF !important;
            color: {ROYAL_BLUE} !important;
            border: 1px solid {BORDER_GREY} !important;
        }}

        /* ---- Select boxes: force white bg / dark grey text ---- */
        div[data-baseweb="select"] > div {{
            background-color: #FFFFFF !important;
            color: {DARK_GREY} !important;
            border: 1px solid {BORDER_GREY} !important;
            border-radius: 8px !important;
            font-size: 1.05rem !important;
            min-height: 3rem !important;
        }}
        div[data-baseweb="select"] span {{
            color: {DARK_GREY} !important;
        }}
        div[data-baseweb="select"] svg {{
            fill: {ROYAL_BLUE} !important;
        }}
        /* dropdown menu popover */
        ul[role="listbox"], div[data-baseweb="popover"] {{
            background-color: #FFFFFF !important;
        }}
        li[role="option"] {{
            background-color: #FFFFFF !important;
            color: {DARK_GREY} !important;
            font-size: 1.0rem !important;
        }}
        li[role="option"]:hover {{
            background-color: #EDF0F8 !important;
        }}

        /* Spacing between form rows */
        div[data-testid="column"] {{
            padding-right: 1.2rem;
            padding-bottom: 0.6rem;
        }}

        /* ---- Submit button ---- */
        div.stButton > button {{
            background-color: {ROYAL_BLUE};
            color: #FFFFFF !important;
            border: none;
            border-radius: 8px;
            padding: 0.75rem 3rem;
            font-weight: 700;
            font-size: 1.1rem;
            margin-top: 1.2rem;
            transition: background-color 0.15s ease-in-out;
        }}
        div.stButton > button:hover {{
            background-color: {ROYAL_BLUE_HOVER};
            color: #FFFFFF !important;
        }}
        div.stButton > button p {{
            color: #FFFFFF !important;
        }}

        /* ---- Result cards ---- */
        .result-box {{
            border-radius: 12px;
            padding: 1.8rem 2rem;
            margin-top: 2rem;
            border: 1px solid {BORDER_GREY};
        }}
        .result-positive {{
            border-left: 6px solid {ROYAL_BLUE};
        }}
        .result-negative {{
            border-left: 6px solid {LIGHT_GREY};
        }}
        .result-title {{
            font-size: 1.4rem;
            font-weight: 700;
            color: {DARK_GREY};
            margin-bottom: 0.4rem;
        }}
        .result-meta {{
            color: {LIGHT_GREY};
            font-size: 0.95rem;
            margin-bottom: 1.1rem;
        }}
        .reason-item {{
            color: {DARK_GREY};
            font-size: 1.02rem;
            margin-bottom: 0.7rem;
            padding: 0.5rem 0 0.5rem 1rem;
            border-left: 3px solid {ROYAL_BLUE};
        }}

        hr {{
            border-top: 2px solid {BORDER_GREY};
        }}

        /* ---- Disease selection cards ---- */
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.disease-card-marker) {{
            border-radius: 14px !important;
            border: 1px solid {BORDER_GREY} !important;
            padding: 0.4rem 0.6rem !important;
            margin-bottom: 1.4rem !important;
            box-shadow: 0 2px 10px rgba(0,0,0,0.04);
            transition: box-shadow 0.15s ease-in-out, border-color 0.15s ease-in-out;
        }}
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.disease-card-marker):hover {{
            border-color: {ROYAL_BLUE} !important;
            box-shadow: 0 4px 16px rgba(42,68,148,0.14);
        }}
        .disease-card-icon {{
            margin-bottom: 0.4rem;
        }}
        .disease-card-title {{
            font-size: 1.5rem;
            font-weight: 700;
            color: #111111;
            margin-bottom: 0.4rem;
        }}
        .disease-card-desc {{
            color: {LIGHT_GREY};
            font-size: 1.0rem;
            line-height: 1.5rem;
            margin-bottom: 0.8rem;
        }}

        /* ---- Centered submit button container ---- */
        .center-btn-wrap div.stButton {{
            display: flex;
            justify-content: center;
        }}

        /* ---- Back / change-condition link button ---- */
        .change-condition-btn button {{
            background-color: transparent !important;
            color: {ROYAL_BLUE} !important;
            border: none !important;
            padding: 0 !important;
            font-weight: 600 !important;
            font-size: 0.95rem !important;
            margin-top: 0 !important;
        }}
        .change-condition-btn button:hover {{
            background-color: transparent !important;
            text-decoration: underline;
        }}
        .change-condition-btn button p {{
            color: {ROYAL_BLUE} !important;
        }}

        footer {{visibility: hidden;}}
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------------------
# PIPELINE LOADING / INFERENCE HELPERS
# --------------------------------------------------------------------------------------
@st.cache_resource
def load_disease_pipeline(disease_name: str):
    models = {
        "rf": joblib.load(f"{disease_name}_rf.pkl"),
        "knn": joblib.load(f"{disease_name}_knn.pkl"),
        "svm": joblib.load(f"{disease_name}_svm.pkl"),
        "nb": joblib.load(f"{disease_name}_nb.pkl"),
    }
    scaler = joblib.load(f"{disease_name}_scaler.pkl")
    with open(f"{disease_name}_thresholds.json", "r") as f:
        thresholds = json.load(f)
    return models, scaler, thresholds


def predict_pipeline(patient_raw: pd.DataFrame, patient_scaled: np.ndarray, models: dict, thresholds: dict):
    """Stage 1 (RF) -> Stage 2 (KNN + SVM + NB, majority vote)."""
    rf_proba = models["rf"].predict_proba(patient_raw)[0][1]

    if rf_proba >= thresholds["rf"]:
        return {
            "prediction": 1,
            "confidence": rf_proba,
            "stage_used": "Stage 1 (Random Forest)",
        }

    knn_proba = models["knn"].predict_proba(patient_scaled)[0][1]
    svm_proba = models["svm"].predict_proba(patient_scaled)[0][1]
    nb_proba = models["nb"].predict_proba(patient_raw)[0][1]

    knn_pred = int(knn_proba >= thresholds["knn"])
    svm_pred = int(svm_proba >= thresholds["svm"])
    nb_pred = int(nb_proba >= thresholds["nb"])

    votes = knn_pred + svm_pred + nb_pred
    final_pred = 1 if votes >= 2 else 0
    avg_confidence = (knn_proba + svm_proba + nb_proba) / 3

    return {
        "prediction": final_pred,
        "confidence": avg_confidence,
        "stage_used": "Stage 2 (Ensemble - Majority Vote)",
        "individual_votes": {"knn": knn_pred, "svm": svm_pred, "nb": nb_pred},
    }


# --------------------------------------------------------------------------------------
# CLINICAL EXPLANATION LOOKUPS  (feature-importance + out-of-range + advice)
# --------------------------------------------------------------------------------------
heart_normal_ranges = {
    "trestbps": (90, 130),
    "chol": (125, 200),
    "thalach": (100, 170),
    "oldpeak": (0, 1),
    "ca": (0, 0),
    "age": (0, 55),
}
heart_clinical_advice = {
    "trestbps": "your resting blood pressure is elevated — consider reducing sodium intake and increasing physical activity.",
    "chol": "your cholesterol is above the healthy range — consider dietary changes and discussing statin therapy with your doctor.",
    "thalach": "your maximum heart rate response is lower than typical — this may reflect reduced cardiovascular fitness.",
    "oldpeak": "your ST depression value suggests notable cardiac stress during exercise — this warrants a cardiology follow-up.",
    "ca": "blocked major vessels were detected — this is a significant finding requiring cardiology follow-up.",
    "age": "age is a non-modifiable risk factor, but managing other risk factors becomes more important as you get older.",
}

ckd_normal_ranges = {
    "bgr": (70, 140),
    "bu": (7, 20),
    "sc": (0.6, 1.3),
    "sod": (135, 145),
    "pot": (3.5, 5.0),
    "hemo": (12, 17),
    "pcv": (36, 50),
    "sg": (1.015, 1.025),
    "al": (0, 0),
}
ckd_clinical_advice = {
    "bgr": "your blood glucose is outside the normal range — managing blood sugar can help protect kidney function.",
    "bu": "your blood urea level is abnormal — this reflects kidney filtration and is worth discussing with a nephrologist.",
    "sc": "your serum creatinine is outside the normal range — a key marker of kidney filtration function.",
    "sod": "your sodium level is abnormal — this can affect fluid balance and kidney function.",
    "pot": "your potassium level is outside the normal range — important to monitor closely with kidney concerns.",
    "hemo": "your hemoglobin is low — reduced kidney function can affect red blood cell production.",
    "pcv": "your packed cell volume is abnormal — may relate to anemia associated with kidney function decline.",
    "sg": "your urine specific gravity is abnormal — reflects how well your kidneys concentrate urine.",
    "al": "protein (albumin) was detected in your urine — an early indicator of kidney damage worth following up.",
}

diabetes_normal_ranges = {
    "stab.glu": (70, 140),
    "chol": (125, 200),
    "ratio": (0, 5),
    "hdl": (40, 100),
    "bp.1s": (90, 130),
    "bp.1d": (60, 85),
    "waist": (0, 40),
    "whr": (0, 0.9),
    "age_waist_interaction": (0, 2000),
    "glucose_fasting_interaction": (0, 140),
}
diabetes_clinical_advice = {
    "stab.glu": "your glucose level is elevated — reducing sugar and refined carbohydrate intake can help lower this.",
    "chol": "your total cholesterol is above the healthy range — consider dietary changes.",
    "ratio": "your cholesterol-to-HDL ratio is elevated — regular exercise can help improve this.",
    "hdl": "your HDL ('good') cholesterol is lower than ideal — regular aerobic exercise can help raise it.",
    "bp.1s": "your systolic blood pressure is elevated — reducing sodium intake may help.",
    "bp.1d": "your diastolic blood pressure is elevated — lifestyle changes like reducing salt can help.",
    "waist": "your waist circumference suggests higher abdominal fat, a known diabetes risk factor.",
    "whr": "your waist-to-hip ratio indicates higher abdominal fat distribution, linked to insulin resistance.",
    "age_waist_interaction": "the combination of your age and waist measurement suggests compounded metabolic risk.",
    "glucose_fasting_interaction": "your fasting glucose reading is elevated — a key diabetes screening indicator.",
}


def generate_explanation(patient_data: pd.DataFrame, rf_model, feature_names, normal_ranges, advice_map):
    importances = pd.Series(rf_model.feature_importances_, index=feature_names).sort_values(ascending=False)
    reasons = []
    for feature in importances.index:
        if feature not in normal_ranges:
            continue
        value = patient_data[feature].values[0]
        low, high = normal_ranges[feature]
        if value < low or value > high:
            advice = advice_map.get(feature, "consider discussing this with your doctor.")
            reasons.append(f"{feature} ({value:.2f}) is outside the typical range — {advice}")
        if len(reasons) >= 3:
            break
    if not reasons:
        reasons = ["No individually abnormal high-impact factors were found, but the overall pattern still suggests risk — please consult a doctor."]
    return reasons


# --------------------------------------------------------------------------------------
# DISEASE FORM CONFIGURATION
# Each entry: (raw_input_key, label, widget_type, kwargs)
# "raw" inputs are what the user actually fills in. Engineered/derived features are
# computed automatically and never shown as separate fields.
# --------------------------------------------------------------------------------------

def label_of(col: str) -> str:
    """Turn a raw column name into a clean, capitalized display label."""
    text = col.replace(".", " ").replace("_", " ").strip()
    text = text[0].upper() + text[1:] if text else text
    return text


HEART_FIELDS = [
    ("age", "number", {"min_value": 1, "max_value": 120, "value": 50, "step": 1}),
    ("sex", "select", {"options": ["Female", "Male"]}),
    ("cp", "select", {"options": ["Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"]}),
    ("trestbps", "number", {"min_value": 60, "max_value": 260, "value": 120, "step": 1}),
    ("chol", "number", {"min_value": 80, "max_value": 700, "value": 200, "step": 1}),
    ("fbs", "select", {"options": ["No (<= 120 mg/dl)", "Yes (> 120 mg/dl)"]}),
    ("restecg", "select", {"options": ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"]}),
    ("thalach", "number", {"min_value": 60, "max_value": 240, "value": 150, "step": 1}),
    ("exang", "select", {"options": ["No", "Yes"]}),
    ("oldpeak", "number", {"min_value": 0.0, "max_value": 10.0, "value": 1.0, "step": 0.1}),
    ("slope", "select", {"options": ["Upsloping", "Flat", "Downsloping"]}),
    ("ca", "select", {"options": ["0", "1", "2", "3"]}),
    ("thal", "select", {"options": ["Normal", "Fixed defect", "Reversible defect"]}),
]

DIABETES_FIELDS = [
    ("stab.glu", "number", {"min_value": 40, "max_value": 500, "value": 100, "step": 1}),
    ("age", "number", {"min_value": 1, "max_value": 120, "value": 45, "step": 1}),
    ("ratio", "number", {"min_value": 0.5, "max_value": 20.0, "value": 4.0, "step": 0.1}),
    ("bp.1s", "number", {"min_value": 60, "max_value": 260, "value": 120, "step": 1}),
    ("hip", "number", {"min_value": 20, "max_value": 80, "value": 40, "step": 1}),
    ("waist", "number", {"min_value": 15, "max_value": 70, "value": 35, "step": 1}),
    ("height", "number", {"min_value": 40, "max_value": 90, "value": 65, "step": 1}),
    ("hdl", "number", {"min_value": 10, "max_value": 150, "value": 50, "step": 1}),
    ("fasting", "select", {"options": ["No", "Yes"]}),
    ("bp.1d", "number", {"min_value": 30, "max_value": 160, "value": 80, "step": 1}),
]

CKD_FIELDS = [
    ("age", "number", {"min_value": 1, "max_value": 110, "value": 45, "step": 1}),
    ("bp (Diastolic)", "select", {"options": ["Normal", "Abnormal"]}),
    ("bp limit", "select", {"options": ["0", "1", "2"]}),
    ("sg", "number", {"min_value": 1.000, "max_value": 1.035, "value": 1.020, "step": 0.001, "format": "%.3f"}),
    ("al", "number", {"min_value": 0.0, "max_value": 5.0, "value": 0.0, "step": 1.0}),
    ("rbc", "select", {"options": ["Normal", "Abnormal"]}),
    ("su", "number", {"min_value": 0.0, "max_value": 5.0, "value": 0.0, "step": 1.0}),
    ("pc", "select", {"options": ["Normal", "Abnormal"]}),
    ("pcc", "select", {"options": ["Not present", "Present"]}),
    ("ba", "select", {"options": ["Not present", "Present"]}),
    ("bgr", "number", {"min_value": 20, "max_value": 500, "value": 120, "step": 1}),
    ("bu", "number", {"min_value": 1, "max_value": 300, "value": 30, "step": 1}),
    ("sod", "number", {"min_value": 100, "max_value": 170, "value": 140, "step": 1}),
    ("sc", "number", {"min_value": 0.1, "max_value": 20.0, "value": 1.0, "step": 0.1}),
    ("pot", "number", {"min_value": 1.0, "max_value": 10.0, "value": 4.5, "step": 0.1}),
    ("hemo", "number", {"min_value": 3.0, "max_value": 20.0, "value": 13.5, "step": 0.1}),
    ("pcv", "number", {"min_value": 10, "max_value": 60, "value": 40, "step": 1}),
    ("rbcc", "number", {"min_value": 1.0, "max_value": 10.0, "value": 4.8, "step": 0.1}),
    ("wbcc", "number", {"min_value": 1000, "max_value": 25000, "value": 8000, "step": 100}),
    ("htn", "select", {"options": ["No", "Yes"]}),
    ("dm", "select", {"options": ["No", "Yes"]}),
    ("cad", "select", {"options": ["No", "Yes"]}),
    ("appet", "select", {"options": ["Good", "Poor"]}),
    ("pe", "select", {"options": ["No", "Yes"]}),
    ("ane", "select", {"options": ["No", "Yes"]}),
]

HEART_ICON_SVG = f"""<svg xmlns="http://www.w3.org/2000/svg" width="34" height="34" viewBox="0 0 24 24"
    fill="none" stroke="{ROYAL_BLUE}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
    <path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"></path>
</svg>"""

PULSE_ICON_SVG = f"""<svg xmlns="http://www.w3.org/2000/svg" width="34" height="34" viewBox="0 0 24 24"
    fill="none" stroke="{ROYAL_BLUE}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
    <path d="M22 12h-4l-3 9L9 3l-3 9H2"></path>
</svg>"""

STETHOSCOPE_ICON_SVG = f"""<svg xmlns="http://www.w3.org/2000/svg" width="34" height="34" viewBox="0 0 24 24"
    fill="none" stroke="{ROYAL_BLUE}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
    <path d="M6 3v6a4 4 0 0 0 8 0V3"></path>
    <path d="M10 13v2a6 6 0 0 0 12 0v-2"></path>
    <circle cx="20" cy="10" r="2"></circle>
    <circle cx="6" cy="3" r="1"></circle>
    <circle cx="14" cy="3" r="1"></circle>
</svg>"""

DISEASE_CONFIG = {
    "Heart Disease": {
        "key": "heart",
        "fields": HEART_FIELDS,
        "feature_order": ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                          "thalach", "exang", "oldpeak", "slope", "ca", "thal"],
        "normal_ranges": heart_normal_ranges,
        "advice_map": heart_clinical_advice,
        "icon": HEART_ICON_SVG,
        "description": "Evaluate risk using cardiac measurements, exercise stress indicators, and angiography data.",
    },
    "Diabetes": {
        "key": "diabetes",
        "fields": DIABETES_FIELDS,
        "feature_order": ["stab.glu", "age_waist_interaction", "glucose_fasting_interaction",
                          "age", "ratio", "whr", "bp.1s", "hip", "waist", "height", "hdl",
                          "fasting", "bp.1d"],
        "normal_ranges": diabetes_normal_ranges,
        "advice_map": diabetes_clinical_advice,
        "icon": PULSE_ICON_SVG,
        "description": "Predict diabetes risk from glucose levels, body composition, and blood pressure readings.",
    },
    "Chronic Kidney Disease": {
        "key": "ckd",
        "fields": CKD_FIELDS,
        "feature_order": ["bp (Diastolic)", "bp limit", "sg", "al", "rbc", "su", "pc", "pcc",
                          "ba", "bgr", "bu", "sod", "sc", "pot", "hemo", "pcv", "rbcc", "wbcc",
                          "htn", "dm", "cad", "appet", "pe", "ane", "age"],
        "normal_ranges": ckd_normal_ranges,
        "advice_map": ckd_clinical_advice,
        "icon": STETHOSCOPE_ICON_SVG,
        "description": "Assess kidney health with urinalysis, blood chemistry, and clinical markers.",
    },
}


def collect_raw_inputs(disease_label: str, fields: list) -> dict:
    """Render the form widgets and collect raw (human-entered) values."""
    raw_values = {}
    cols = st.columns(2)
    for i, (col_name, widget_type, kwargs) in enumerate(fields):
        target_col = cols[i % 2]
        with target_col:
            display_label = label_of(col_name)
            widget_key = f"{disease_label}_{col_name}"
            if widget_type == "number":
                raw_values[col_name] = st.number_input(display_label, key=widget_key, **kwargs)
            elif widget_type == "select":
                raw_values[col_name] = st.selectbox(display_label, kwargs["options"], key=widget_key)
    return raw_values


def build_model_row(disease_key: str, raw_values: dict, feature_order: list) -> pd.DataFrame:
    """Convert raw form values into the exact numeric row the model expects."""
    row = {}

    if disease_key == "heart":
        row["age"] = raw_values["age"]
        row["sex"] = 1 if raw_values["sex"] == "Male" else 0
        row["cp"] = ["Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"].index(raw_values["cp"])
        row["trestbps"] = raw_values["trestbps"]
        row["chol"] = raw_values["chol"]
        row["fbs"] = 1 if raw_values["fbs"].startswith("Yes") else 0
        row["restecg"] = ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"].index(raw_values["restecg"])
        row["thalach"] = raw_values["thalach"]
        row["exang"] = 1 if raw_values["exang"] == "Yes" else 0
        row["oldpeak"] = raw_values["oldpeak"]
        row["slope"] = ["Upsloping", "Flat", "Downsloping"].index(raw_values["slope"])
        row["ca"] = int(raw_values["ca"])
        row["thal"] = ["Normal", "Fixed defect", "Reversible defect"].index(raw_values["thal"])

    elif disease_key == "diabetes":
        stab_glu = raw_values["stab.glu"]
        age = raw_values["age"]
        waist = raw_values["waist"]
        hip = raw_values["hip"]
        fasting = 1 if raw_values["fasting"] == "Yes" else 0

        row["stab.glu"] = stab_glu
        row["age_waist_interaction"] = age * waist
        row["glucose_fasting_interaction"] = stab_glu * fasting
        row["age"] = age
        row["ratio"] = raw_values["ratio"]
        row["whr"] = waist / hip if hip else 0
        row["bp.1s"] = raw_values["bp.1s"]
        row["hip"] = hip
        row["waist"] = waist
        row["height"] = raw_values["height"]
        row["hdl"] = raw_values["hdl"]
        row["fasting"] = fasting
        row["bp.1d"] = raw_values["bp.1d"]

    elif disease_key == "ckd":
        row["bp (Diastolic)"] = 1 if raw_values["bp (Diastolic)"] == "Abnormal" else 0
        row["bp limit"] = int(raw_values["bp limit"])
        row["sg"] = raw_values["sg"]
        row["al"] = raw_values["al"]
        row["rbc"] = 1 if raw_values["rbc"] == "Abnormal" else 0
        row["su"] = raw_values["su"]
        row["pc"] = 1 if raw_values["pc"] == "Abnormal" else 0
        row["pcc"] = 1 if raw_values["pcc"] == "Present" else 0
        row["ba"] = 1 if raw_values["ba"] == "Present" else 0
        row["bgr"] = raw_values["bgr"]
        row["bu"] = raw_values["bu"]
        row["sod"] = raw_values["sod"]
        row["sc"] = raw_values["sc"]
        row["pot"] = raw_values["pot"]
        row["hemo"] = raw_values["hemo"]
        row["pcv"] = raw_values["pcv"]
        row["rbcc"] = raw_values["rbcc"]
        row["wbcc"] = raw_values["wbcc"]
        row["htn"] = 1 if raw_values["htn"] == "Yes" else 0
        row["dm"] = 1 if raw_values["dm"] == "Yes" else 0
        row["cad"] = 1 if raw_values["cad"] == "Yes" else 0
        row["appet"] = 1 if raw_values["appet"] == "Poor" else 0
        row["pe"] = 1 if raw_values["pe"] == "Yes" else 0
        row["ane"] = 1 if raw_values["ane"] == "Yes" else 0
        row["age"] = raw_values["age"]

    return pd.DataFrame([row])[feature_order]


# --------------------------------------------------------------------------------------
# MAIN APP
# --------------------------------------------------------------------------------------
st.markdown('<div class="app-title">Disease Risk Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Select a condition, enter the requested values, and get a risk assessment.</div>',
    unsafe_allow_html=True,
)

if "selected_disease" not in st.session_state:
    st.session_state.selected_disease = None


def render_disease_cards():
    """Card-based picker shown instead of a dropdown."""
    for label, cfg in DISEASE_CONFIG.items():
        with st.container(border=True):
            st.markdown('<span class="disease-card-marker"></span>', unsafe_allow_html=True)
            st.markdown(f'<div class="disease-card-icon">{cfg["icon"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="disease-card-title">{label}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="disease-card-desc">{cfg["description"]}</div>', unsafe_allow_html=True)
            if st.button("Select", key=f"pick_{cfg['key']}", use_container_width=True):
                st.session_state.selected_disease = label
                st.rerun()


if st.session_state.selected_disease is None:
    render_disease_cards()
else:
    disease_label = st.session_state.selected_disease
    config = DISEASE_CONFIG[disease_label]

    st.markdown('<div class="change-condition-btn">', unsafe_allow_html=True)
    if st.button("← Choose a different condition", key="change_condition"):
        st.session_state.selected_disease = None
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(f'<div class="section-header">{disease_label} — Patient details</div>', unsafe_allow_html=True)
    raw_values = collect_raw_inputs(disease_label, config["fields"])

    st.markdown('<div class="center-btn-wrap">', unsafe_allow_html=True)
    submitted = st.button("Submit")
    st.markdown('</div>', unsafe_allow_html=True)

    if submitted:
        try:
            models, scaler, thresholds = load_disease_pipeline(config["key"])
        except FileNotFoundError as e:
            st.error(f"Could not find a required model file: {e}. Make sure all .pkl / .json files are in the app folder.")
            st.stop()

        patient_row = build_model_row(config["key"], raw_values, config["feature_order"])
        patient_scaled = scaler.transform(patient_row)

        result = predict_pipeline(patient_row, patient_scaled, models, thresholds)

        is_positive = result["prediction"] == 1
        box_class = "result-positive" if is_positive else "result-negative"
        verdict_text = "Positive risk indicators found" if is_positive else "No significant risk indicators found"

        st.markdown(
            f"""
            <div class="result-box {box_class}">
                <div class="result-title">{disease_label}: {verdict_text}</div>
                <div class="result-meta">Confidence: {result['confidence']*100:.1f}%</div>
            """,
            unsafe_allow_html=True,
        )

        if is_positive:
            reasons = generate_explanation(
                patient_row, models["rf"], config["feature_order"],
                config["normal_ranges"], config["advice_map"]
            )
            st.markdown('<div style="margin-top:0.4rem; font-weight:600;">Likely contributing factors:</div>', unsafe_allow_html=True)
            for r in reasons:
                st.markdown(f'<div class="reason-item">{r}</div>', unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            '<div class="result-meta" style="margin-top:1rem;">'
            'This tool provides a statistical risk estimate for educational purposes only and is not a medical diagnosis. '
            'Please consult a qualified healthcare professional.'
            '</div>',
            unsafe_allow_html=True,
        )
