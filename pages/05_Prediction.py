"""
pages/05_Prediction.py — Single employee attrition prediction.
"""

import streamlit as st
import pandas as pd
import numpy as np

from ml.train_pipeline import load_metrics, PIPELINE_PATH
from ml.predict import get_pipeline, predict_single, get_feature_metadata
from ml.data_loader import load_raw_dataset, dataset_exists
import os

st.markdown("# 🔮 Employee Attrition Prediction")
st.caption("Enter employee details to predict attrition risk. Predictions are estimates, not certainties.")

if not os.path.isfile(PIPELINE_PATH):
    st.error("Model not trained yet. Please go to **Model Training** first.")
    st.stop()

@st.cache_resource(show_spinner=False)
def _get_pipeline_cached():
    return get_pipeline()

pipeline, meta = _get_pipeline_cached()
feature_info = get_feature_metadata(meta)
feature_names = feature_info["feature_names"]
cat_cols = feature_info["categorical_cols"]
num_cols = feature_info["numeric_cols"]

# Get valid categories from the actual dataset
@st.cache_data
def get_categories():
    if not dataset_exists():
        return {}
    df = load_raw_dataset()
    cats = {}
    for c in cat_cols:
        if c in df.columns:
            cats[c] = sorted(df[c].dropna().unique().tolist())
    return cats

categories = get_categories()

# ── Input form ─────────────────────────────────────────────────────────────────
st.markdown("### Employee Details")
st.info("Fill in employee attributes. All fields are required for accurate prediction.")

input_data = {}

with st.form("prediction_form"):
    # Split into columns for better layout
    col_a, col_b, col_c = st.columns(3)

    for i, feat in enumerate(feature_names):
        col = [col_a, col_b, col_c][i % 3]
        with col:
            if feat in cat_cols:
                options = categories.get(feat, ["Yes", "No"])
                input_data[feat] = st.selectbox(feat, options, key=f"inp_{feat}")
            else:
                # Determine reasonable defaults from feature name
                defaults = {
                    "Age": 35, "MonthlyIncome": 5000, "DailyRate": 800,
                    "HourlyRate": 65, "MonthlyRate": 14000, "DistanceFromHome": 5,
                    "NumCompaniesWorked": 3, "PercentSalaryHike": 15,
                    "TotalWorkingYears": 10, "TrainingTimesLastYear": 3,
                    "YearsAtCompany": 5, "YearsInCurrentRole": 3,
                    "YearsSinceLastPromotion": 2, "YearsWithCurrManager": 3,
                }
                mins = {
                    "Age": 18, "MonthlyIncome": 1009, "PerformanceRating": 1,
                    "JobSatisfaction": 1, "WorkLifeBalance": 1, "Education": 1,
                    "EnvironmentSatisfaction": 1, "JobInvolvement": 1, "JobLevel": 1,
                    "RelationshipSatisfaction": 1, "StockOptionLevel": 0,
                    "NumCompaniesWorked": 0, "TrainingTimesLastYear": 0,
                    "YearsAtCompany": 0, "YearsInCurrentRole": 0,
                    "YearsSinceLastPromotion": 0, "YearsWithCurrManager": 0,
                }
                maxs = {
                    "Age": 65, "MonthlyIncome": 200000, "PerformanceRating": 4,
                    "JobSatisfaction": 4, "WorkLifeBalance": 4, "Education": 5,
                    "EnvironmentSatisfaction": 4, "JobInvolvement": 4, "JobLevel": 5,
                    "RelationshipSatisfaction": 4, "StockOptionLevel": 3,
                    "NumCompaniesWorked": 9, "TrainingTimesLastYear": 6,
                    "YearsAtCompany": 40, "YearsInCurrentRole": 18,
                    "YearsSinceLastPromotion": 15, "YearsWithCurrManager": 17,
                }
                dv = defaults.get(feat, 1)
                mn = mins.get(feat, 0)
                mx = maxs.get(feat, 100)
                if mx - mn > 10:
                    input_data[feat] = st.number_input(feat, min_value=mn, max_value=mx, value=dv, key=f"inp_{feat}")
                else:
                    input_data[feat] = st.slider(feat, min_value=mn, max_value=mx, value=dv, key=f"inp_{feat}")

    col_sub, col_rst = st.columns([2, 1])
    with col_sub:
        submitted = st.form_submit_button("🔮 Predict Attrition Risk", type="primary", use_container_width=True)
    with col_rst:
        reset = st.form_submit_button("🔄 Reset Form", use_container_width=True)

if reset:
    for key in list(st.session_state.keys()):
        if key.startswith("inp_"):
            del st.session_state[key]
    st.rerun()

# ── Result ─────────────────────────────────────────────────────────────────────
if submitted:
    try:
        result = predict_single(input_data)
        st.session_state.prediction_made = True
        st.session_state.last_prediction = result
        st.session_state.last_input = input_data

        st.divider()
        st.markdown("### 📊 Prediction Result")

        pred_color = "#ef4444" if result["predicted_class"] == 1 else "#10b981"
        st.markdown(f"""
        <div style="background:white;border-radius:14px;padding:24px;border:2px solid {pred_color};text-align:center;margin-bottom:16px">
            <div style="font-size:40px">{result['risk_level'].split()[0]}</div>
            <div style="font-size:22px;font-weight:700;color:{pred_color}">{result['label']}</div>
            <div style="font-size:14px;color:#64748b;margin-top:4px">{result['risk_level']}</div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        col1.metric("Probability of Leaving", f"{result['probability_leave']}%")
        col2.metric("Probability of Staying", f"{result['probability_stay']}%")

        # Confidence bar
        st.markdown(f"**Leave Probability: {result['probability_leave']}%**")
        st.progress(result["probability_leave"] / 100)

        st.warning("""
        ⚠️ **Disclaimer:** This prediction is a statistical estimate based on historical patterns. 
        It should be used as one input among many HR considerations, not as a definitive conclusion about any individual employee.
        """)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
