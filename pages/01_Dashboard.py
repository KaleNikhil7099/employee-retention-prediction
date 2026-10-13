"""
pages/01_Dashboard.py — RetentionAI Dashboard Overview
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from ml.data_loader import load_raw_dataset, get_dataset_info, get_dept_attrition, get_role_attrition, dataset_exists
from ml.train_pipeline import load_metrics

st.markdown("# 🏠 Dashboard Overview")
st.caption("Real-time metrics and analytics from the IBM HR dataset.")

# ── Dataset check ─────────────────────────────────────────────────────────────
if not dataset_exists():
    st.warning("""
    📂 **Dataset not found!**
    
    Please download the IBM HR Analytics dataset from Kaggle and place it at:
    ```
    data/WA_Fn-UseC_-HR-Employee-Attrition.csv
    ```
    [Download from Kaggle →](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
    """)
    st.stop()

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading dataset...")
def _load():
    df = load_raw_dataset()
    info = get_dataset_info(df)
    return df, info

df, info = _load()
st.session_state.dataset_loaded = True

metrics = load_metrics()
if metrics:
    st.session_state.model_trained = True

# ── KPI Cards ─────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("👥 Total Employees", f"{info['total_rows']:,}")
col2.metric("🚪 Employees Left", f"{info['attrition_yes']:,}")
col3.metric("✅ Employees Stayed", f"{info['attrition_no']:,}")
col4.metric("📉 Attrition Rate", f"{info['attrition_rate']}%")

st.divider()

# ── Model metrics banner ───────────────────────────────────────────────────────
if metrics:
    best = metrics.get("best_model", "N/A")
    bm = metrics["models"].get(best, {})
    st.markdown(f"#### 🏆 Best Model: **{best}**")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{bm.get('accuracy', 0):.1%}")
    m2.metric("F1-Score", f"{bm.get('f1_score', 0):.3f}")
    m3.metric("Precision", f"{bm.get('precision', 0):.3f}")
    m4.metric("ROC-AUC", f"{bm.get('roc_auc', 0):.3f}")
    st.divider()
else:
    st.info("ℹ️ Model not yet trained. Go to **Model Training** to train models.")

# ── Charts ─────────────────────────────────────────────────────────────────────
col_left, col_right = st.columns(2)

with col_left:
    st.markdown("#### Attrition Distribution")
    attrition_df = pd.DataFrame({
        "Status": ["Stayed", "Left"],
        "Count": [info['attrition_no'], info['attrition_yes']]
    })
    fig_pie = px.pie(
        attrition_df, names="Status", values="Count",
        color_discrete_sequence=["#0d9488", "#ef4444"],
        hole=0.45
    )
    fig_pie.update_layout(
        margin=dict(t=20, b=20, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=True,
        font=dict(family="Inter")
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with col_right:
    st.markdown("#### Attrition Rate by Department")
    dept_df = get_dept_attrition(df)
    if not dept_df.empty:
        fig_dept = px.bar(
            dept_df, x="Attrition Rate (%)", y="Department",
            orientation="h",
            color="Attrition Rate (%)",
            color_continuous_scale=["#0d9488", "#ef4444"],
        )
        fig_dept.update_layout(
            margin=dict(t=20, b=20, l=20, r=20),
            paper_bgcolor="rgba(0,0,0,0)",
            coloraxis_showscale=False,
            font=dict(family="Inter")
        )
        st.plotly_chart(fig_dept, use_container_width=True)

# ── Job role analysis ─────────────────────────────────────────────────────────
st.markdown("#### Attrition Rate by Job Role")
role_df = get_role_attrition(df)
if not role_df.empty:
    fig_role = px.bar(
        role_df, x="Job Role", y="Attrition Rate (%)",
        color="Attrition Rate (%)",
        color_continuous_scale=["#0d9488", "#f59e0b", "#ef4444"],
    )
    fig_role.update_layout(
        margin=dict(t=20, b=80, l=20, r=20),
        xaxis_tickangle=-35,
        paper_bgcolor="rgba(0,0,0,0)",
        coloraxis_showscale=False,
        font=dict(family="Inter")
    )
    st.plotly_chart(fig_role, use_container_width=True)

# ── Workflow status ────────────────────────────────────────────────────────────
st.divider()
st.markdown("#### 🗺️ Workflow Progress")
workflow_steps = [
    ("Dataset Loaded", st.session_state.get("dataset_loaded", False)),
    ("Preprocessing Viewed", st.session_state.get("preprocessing_done", False)),
    ("Model Trained", st.session_state.get("model_trained", False)),
    ("Prediction Made", st.session_state.get("prediction_made", False)),
]
cols = st.columns(len(workflow_steps))
for col, (label, done) in zip(cols, workflow_steps):
    icon = "✅" if done else "⏳"
    col.markdown(f"**{icon} {label}**")
