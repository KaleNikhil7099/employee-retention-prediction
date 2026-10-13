"""
pages/04_Model_Training.py — Train models and view evaluation results.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import json
import os

from ml.train_pipeline import train_all_models, load_metrics, METRICS_PATH
from ml.data_loader import dataset_exists

st.markdown("# 🧪 Model Training & Evaluation")
st.caption("Train Logistic Regression, Random Forest, and Gradient Boosting on the IBM HR dataset.")

if not dataset_exists():
    st.error("Dataset not found. Please add the IBM HR CSV to `data/` first.")
    st.stop()

# ── Training ──────────────────────────────────────────────────────────────────
metrics = load_metrics()

col_train, col_status = st.columns([2, 1])
with col_train:
    if st.button("🚀 Train All Models", type="primary", use_container_width=True):
        progress_bar = st.progress(0)
        status_text = st.empty()

        def _progress(step, total, msg):
            progress_bar.progress(step / total)
            status_text.markdown(f"**{msg}**")

        try:
            with st.spinner("Training models — this may take 1-2 minutes..."):
                result = train_all_models(progress_callback=_progress)
            st.session_state.model_trained = True
            st.success(f"✅ Training complete! Best model: **{result['best_model']}**")
            metrics = result
            st.rerun()
        except FileNotFoundError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Training failed: {e}")

with col_status:
    if metrics:
        st.success("✅ Trained model found")
    else:
        st.warning("⏳ Not yet trained")

# ── Show results ──────────────────────────────────────────────────────────────
if not metrics:
    st.info("Click **Train All Models** to begin. Training trains 3 models and selects the best by F1-score.")
    st.stop()

st.session_state.model_trained = True
best_name = metrics.get("best_model", "")

st.divider()
st.markdown(f"### 🏆 Best Model: **{best_name}**")

# Model comparison table
rows = []
for name, m in metrics["models"].items():
    rows.append({
        "Model": ("🥇 " if name == best_name else "") + name,
        "Accuracy": f"{m['accuracy']:.4f}",
        "Precision": f"{m['precision']:.4f}",
        "Recall": f"{m['recall']:.4f}",
        "F1-Score": f"{m['f1_score']:.4f}",
        "ROC-AUC": f"{m['roc_auc']:.4f}",
    })
comparison_df = pd.DataFrame(rows)
st.dataframe(comparison_df, use_container_width=True, hide_index=True)

# ── Per-model detailed metrics ─────────────────────────────────────────────────
st.divider()
selected = st.selectbox("Select model to inspect", list(metrics["models"].keys()), index=list(metrics["models"].keys()).index(best_name))
m = metrics["models"][selected]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", f"{m['accuracy']:.4f}")
col2.metric("F1-Score", f"{m['f1_score']:.4f}")
col3.metric("Precision", f"{m['precision']:.4f}")
col4.metric("Recall", f"{m['recall']:.4f}")
st.metric("ROC-AUC", f"{m['roc_auc']:.4f}")

# Confusion matrix
cm = m.get("confusion_matrix", [[0,0],[0,0]])
st.markdown("#### Confusion Matrix")
labels = ["Stayed (0)", "Left (1)"]
fig_cm = ff.create_annotated_heatmap(
    z=cm, x=labels, y=labels,
    colorscale="Teal", showscale=True
)
fig_cm.update_layout(
    title_text=f"Confusion Matrix — {selected}",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter")
)
st.plotly_chart(fig_cm, use_container_width=True)

# Class report
cr = m.get("classification_report", {})
if cr:
    st.markdown("#### Classification Report")
    cr_rows = []
    for label, vals in cr.items():
        if isinstance(vals, dict):
            cr_rows.append({
                "Class": label,
                "Precision": round(vals.get("precision", 0), 4),
                "Recall": round(vals.get("recall", 0), 4),
                "F1-Score": round(vals.get("f1-score", 0), 4),
                "Support": int(vals.get("support", 0)),
            })
    st.dataframe(pd.DataFrame(cr_rows), hide_index=True, use_container_width=True)

# Class imbalance note
st.info("""
**ℹ️ Class Imbalance Note:** The IBM HR dataset has ~16% attrition rate. All models use `class_weight='balanced'` 
to prevent the classifier from simply predicting "Stay" for everyone. Recall and F1-Score are more meaningful 
metrics than raw accuracy for imbalanced datasets.
""")
