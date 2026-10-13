"""
ml/predict.py — Prediction helpers for RetentionAI.
Loads saved pipeline; provides single and batch inference.
"""

import os
import pandas as pd
import numpy as np
import streamlit as st
from ml.train_pipeline import load_pipeline


@st.cache_resource(show_spinner=False)
def get_pipeline():
    """Cached pipeline loader."""
    return load_pipeline()


def predict_single(input_dict: dict) -> dict:
    """
    Predict attrition for a single employee.
    input_dict: {feature_name: value, ...}
    Returns: {predicted_class, probability_leave, probability_stay, risk_level}
    """
    pipeline, meta = get_pipeline()
    feature_names = meta["feature_names"]

    # Build DataFrame row
    row = {f: input_dict.get(f, np.nan) for f in feature_names}
    df_input = pd.DataFrame([row])[feature_names]

    pred = int(pipeline.predict(df_input)[0])
    prob = None
    if hasattr(pipeline, "predict_proba"):
        prob = pipeline.predict_proba(df_input)[0]

    prob_leave = float(prob[1]) if prob is not None else float(pred)
    prob_stay = float(prob[0]) if prob is not None else 1.0 - float(pred)

    if prob_leave >= 0.70:
        risk_level = "🔴 High Risk"
    elif prob_leave >= 0.40:
        risk_level = "🟡 Medium Risk"
    else:
        risk_level = "🟢 Low Risk"

    return {
        "predicted_class": pred,
        "label": "Likely to Leave" if pred == 1 else "Likely to Stay",
        "probability_leave": round(prob_leave * 100, 1),
        "probability_stay": round(prob_stay * 100, 1),
        "risk_level": risk_level,
    }


def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Predict attrition for multiple employees.
    df: DataFrame with employee features (no target column required).
    Returns df with additional columns: Predicted_Attrition, Probability_Leave_%, Risk_Level
    """
    pipeline, meta = get_pipeline()
    feature_names = meta["feature_names"]

    # Find which required features are present
    missing = [f for f in feature_names if f not in df.columns]
    extra = [c for c in df.columns if c not in feature_names and c != "Attrition"]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    X = df[feature_names].copy()
    preds = pipeline.predict(X)
    probs = None
    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba(X)[:, 1]

    result = df.copy()
    result["Predicted_Attrition"] = ["Yes" if p == 1 else "No" for p in preds]
    if probs is not None:
        result["Probability_Leave_%"] = (probs * 100).round(1)
        result["Risk_Level"] = pd.cut(
            probs * 100,
            bins=[-1, 40, 70, 101],
            labels=["Low Risk", "Medium Risk", "High Risk"]
        ).astype(str)

    return result, missing, extra


def get_feature_metadata(meta: dict) -> dict:
    """Return info about features for building input forms."""
    return {
        "feature_names": meta.get("feature_names", []),
        "categorical_cols": meta.get("categorical_cols", []),
        "numeric_cols": meta.get("numeric_cols", []),
    }
