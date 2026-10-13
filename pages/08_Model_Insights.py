"""
pages/08_Model_Insights.py — Feature importance and model analytics.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

from ml.train_pipeline import load_metrics, PIPELINE_PATH
from ml.data_loader import load_raw_dataset, dataset_exists

st.markdown("# 📊 Model Insights")
st.caption("Feature importance and attrition analytics from the trained model.")

metrics = load_metrics()
if not metrics:
    st.error("Model not trained yet. Go to **Model Training** to train your models.")
    st.stop()

tab1, tab2, tab3 = st.tabs(["🎯 Feature Importance", "📈 Model Comparison", "🔬 Dataset Analytics"])

with tab1:
    st.markdown("#### Top Contributing Features (Best Model)")
    fi = metrics.get("feature_importances", {})
    if fi:
        fi_df = pd.DataFrame(list(fi.items()), columns=["Feature", "Importance"])
        fi_df = fi_df.sort_values("Importance", ascending=False).head(20)
        # Clean feature names (remove prefix from ColumnTransformer)
        fi_df["Feature"] = fi_df["Feature"].str.replace("num__", "").str.replace("cat__", "")
        fig_fi = px.bar(
            fi_df, x="Importance", y="Feature", orientation="h",
            color="Importance",
            color_continuous_scale=["#0891b2", "#0d9488", "#10b981"],
            title=f"Top 20 Features — {metrics.get('best_model', 'Best Model')}"
        )
        fig_fi.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", yaxis=dict(autorange="reversed"),
            coloraxis_showscale=False, font=dict(family="Inter")
        )
        st.plotly_chart(fig_fi, use_container_width=True)
        st.caption("Feature importance reflects how much each feature contributes to the model's predictions. Higher = more influential.")
    else:
        st.info("Feature importance data not available for this model type.")

with tab2:
    st.markdown("#### Model Performance Comparison")
    model_data = metrics.get("models", {})
    comparison = []
    for name, m in model_data.items():
        comparison.append({
            "Model": name,
            "Accuracy": m["accuracy"],
            "Precision": m["precision"],
            "Recall": m["recall"],
            "F1-Score": m["f1_score"],
            "ROC-AUC": m["roc_auc"],
        })
    df_comp = pd.DataFrame(comparison)

    metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    df_melted = df_comp.melt(id_vars="Model", value_vars=metrics_to_plot, var_name="Metric", value_name="Score")

    fig_comp = px.bar(
        df_melted, x="Metric", y="Score", color="Model",
        barmode="group",
        color_discrete_sequence=["#0891b2", "#0d9488", "#f59e0b"],
        title="Model Performance Comparison"
    )
    fig_comp.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", yaxis_range=[0, 1],
        font=dict(family="Inter")
    )
    st.plotly_chart(fig_comp, use_container_width=True)

    st.dataframe(df_comp.set_index("Model").round(4), use_container_width=True)

    # Training info
    st.markdown("**Training Details:**")
    c1, c2 = st.columns(2)
    c1.metric("Training Samples", f"{metrics.get('training_samples', 'N/A'):,}" if isinstance(metrics.get('training_samples'), int) else "N/A")
    c2.metric("Test Samples", f"{metrics.get('test_samples', 'N/A'):,}" if isinstance(metrics.get('test_samples'), int) else "N/A")

with tab3:
    if not dataset_exists():
        st.warning("Dataset not found. Load the IBM HR CSV first.")
    else:
        @st.cache_data
        def _load(): return load_raw_dataset()
        df = _load()

        # Correlation between numeric features and attrition
        st.markdown("#### Numeric Feature Correlation with Attrition")
        df_num = df.copy()
        df_num["Attrition_Num"] = (df_num["Attrition"].str.lower() == "yes").astype(int)
        num_cols = df.select_dtypes(include="number").columns.tolist()
        corr = df_num[num_cols + ["Attrition_Num"]].corr()["Attrition_Num"].drop("Attrition_Num").sort_values()
        corr_df = corr.reset_index()
        corr_df.columns = ["Feature", "Correlation"]
        fig_corr = px.bar(
            corr_df, x="Correlation", y="Feature", orientation="h",
            color="Correlation", color_continuous_scale="RdYlGn",
            title="Feature Correlation with Attrition (Pearson)"
        )
        fig_corr.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", yaxis=dict(autorange="reversed"),
            font=dict(family="Inter")
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        st.caption("Positive = associated with leaving, Negative = associated with staying.")

        # Income vs Attrition scatter
        if "MonthlyIncome" in df.columns and "TotalWorkingYears" in df.columns:
            st.markdown("#### Monthly Income vs Total Working Years")
            fig_scatter = px.scatter(
                df, x="TotalWorkingYears", y="MonthlyIncome", color="Attrition",
                color_discrete_map={"Yes": "#ef4444", "No": "#0d9488"},
                opacity=0.6, title="Income vs Experience by Attrition"
            )
            fig_scatter.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter"))
            st.plotly_chart(fig_scatter, use_container_width=True)
