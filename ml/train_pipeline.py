"""
ml/train_pipeline.py — Train and evaluate multiple classifiers on the IBM HR dataset.

Trains: Logistic Regression, Random Forest, Gradient Boosting.
Handles class imbalance. Saves the best full pipeline to models/pipeline.pkl.
Saves all metrics to models/metrics.json.

Run standalone: python ml/train_pipeline.py
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report
)

# Add project root to path when running standalone
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from ml.data_loader import load_raw_dataset, prepare_features

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
PIPELINE_PATH = os.path.join(MODELS_DIR, "pipeline.pkl")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")


def build_preprocessor(categorical_cols: list, numeric_cols: list) -> ColumnTransformer:
    """Build a ColumnTransformer: OrdinalEncoder for cats, StandardScaler for nums."""
    transformers = []
    if numeric_cols:
        transformers.append(("num", StandardScaler(), numeric_cols))
    if categorical_cols:
        transformers.append(
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), categorical_cols)
        )
    return ColumnTransformer(transformers=transformers, remainder="drop")


def _evaluate_model(pipeline, X_test, y_test) -> dict:
    y_pred = pipeline.predict(X_test)
    y_prob = None
    if hasattr(pipeline, "predict_proba"):
        y_prob = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy":  round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall":    round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1_score":  round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "roc_auc":   round(float(roc_auc_score(y_test, y_prob)) if y_prob is not None else 0.0, 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
    }
    return metrics


def train_all_models(progress_callback=None) -> dict:
    """
    Full training run. Returns dict with best model info and all metrics.
    progress_callback(step, total, message) — optional for Streamlit progress bars.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    def _prog(step, total, msg):
        if progress_callback:
            progress_callback(step, total, msg)
        else:
            print(f"[{step}/{total}] {msg}")

    _prog(1, 8, "Loading dataset...")
    df = load_raw_dataset()
    X, y, categorical_cols, numeric_cols = prepare_features(df)

    _prog(2, 8, "Splitting train/test (80/20, stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    _prog(3, 8, "Building preprocessing pipeline...")
    preprocessor = build_preprocessor(categorical_cols, numeric_cols)

    classifiers = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=42, C=0.1
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1, max_depth=12
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, random_state=42, learning_rate=0.05, max_depth=5, subsample=0.8
        ),
    }

    results = {}
    best_name = None
    best_f1 = -1
    best_pipeline = None

    total_steps = 8
    step = 4

    for name, clf in classifiers.items():
        _prog(step, total_steps, f"Training {name}...")
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", clf)
        ])
        pipeline.fit(X_train, y_train)
        metrics = _evaluate_model(pipeline, X_test, y_test)
        results[name] = metrics

        if metrics["f1_score"] > best_f1:
            best_f1 = metrics["f1_score"]
            best_name = name
            best_pipeline = pipeline

        step += 1

    _prog(step, total_steps, "Computing feature importances...")
    feature_names_out = []
    try:
        feature_names_out = list(
            best_pipeline.named_steps["preprocessor"].get_feature_names_out()
        )
    except Exception:
        feature_names_out = numeric_cols + categorical_cols

    feature_importances = {}
    clf_step = best_pipeline.named_steps["classifier"]
    if hasattr(clf_step, "feature_importances_"):
        imps = clf_step.feature_importances_
        feature_importances = dict(zip(feature_names_out, [round(float(v), 6) for v in imps]))
    elif hasattr(clf_step, "coef_"):
        imps = np.abs(clf_step.coef_[0])
        feature_importances = dict(zip(feature_names_out, [round(float(v), 6) for v in imps]))

    step += 1
    _prog(step, total_steps, "Saving pipeline and metrics...")

    # Save full output
    output = {
        "best_model": best_name,
        "models": results,
        "feature_importances": feature_importances,
        "feature_names": list(X.columns),
        "categorical_cols": categorical_cols,
        "numeric_cols": numeric_cols,
        "training_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "class_distribution": {
            "train_attrition_rate": round(float(y_train.mean() * 100), 2),
            "test_attrition_rate": round(float(y_test.mean() * 100), 2),
        },
    }

    joblib.dump({"pipeline": best_pipeline, "meta": output}, PIPELINE_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump(output, f, indent=2)

    _prog(8, 8, f"Done! Best model: {best_name} (F1={best_f1:.4f})")
    return output


def load_pipeline():
    """Load saved pipeline. Returns (pipeline, meta) or raises FileNotFoundError."""
    if not os.path.isfile(PIPELINE_PATH):
        raise FileNotFoundError(
            "Model pipeline not found. Please go to 'Model Training' and train the models first."
        )
    data = joblib.load(PIPELINE_PATH)
    return data["pipeline"], data["meta"]


def load_metrics() -> dict | None:
    """Load saved metrics JSON."""
    if not os.path.isfile(METRICS_PATH):
        return None
    with open(METRICS_PATH) as f:
        return json.load(f)


if __name__ == "__main__":
    print("Starting RetentionAI model training...")
    result = train_all_models()
    print("\n=== Training Complete ===")
    print(f"Best model: {result['best_model']}")
    for name, m in result["models"].items():
        print(f"  {name}: Acc={m['accuracy']:.4f} F1={m['f1_score']:.4f} ROC-AUC={m['roc_auc']:.4f}")
    print(f"Pipeline saved to: {PIPELINE_PATH}")
