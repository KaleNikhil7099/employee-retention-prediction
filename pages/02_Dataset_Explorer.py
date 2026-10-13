"""
pages/02_Dataset_Explorer.py — IBM HR Dataset inspection and EDA.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

from ml.data_loader import load_raw_dataset, get_dataset_info, dataset_exists

st.markdown("# 📦 Dataset Explorer")
st.caption("IBM HR Analytics Employee Attrition & Performance Dataset (fictional dataset — not real IBM employee records).")

if not dataset_exists():
    st.error("""
    **Dataset file not found.**
    
    Please download `WA_Fn-UseC_-HR-Employee-Attrition.csv` from Kaggle and place it in the `data/` folder:
    
    🔗 https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset
    """)
    st.stop()

@st.cache_data
def _load():
    df = load_raw_dataset()
    info = get_dataset_info(df)
    return df, info

df, info = _load()

# ── Overview metrics ──────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("📋 Rows", f"{info['total_rows']:,}")
col2.metric("📐 Columns", f"{info['total_cols']}")
col3.metric("❓ Missing Values", f"{info['missing_values']:,}")
col4.metric("🔁 Duplicate Rows", f"{info['duplicate_rows']:,}")

st.divider()

tab1, tab2, tab3, tab4 = st.tabs(["📋 Preview", "📊 Column Info", "📈 Distributions", "🏢 Department Analysis"])

with tab1:
    st.markdown(f"**Showing first 10 rows of {info['total_rows']:,} total records**")
    st.dataframe(df.head(10), use_container_width=True)

with tab2:
    # Column info table
    col_info = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values,
        "Non-Null": df.count().values,
        "Null Count": df.isnull().sum().values,
        "Unique Values": df.nunique().values,
        "Sample Value": [df[c].dropna().iloc[0] if not df[c].dropna().empty else "" for c in df.columns]
    })
    st.dataframe(col_info, use_container_width=True)

    st.markdown("**Attrition Target Distribution**")
    attrition_counts = df["Attrition"].value_counts()
    c1, c2 = st.columns(2)
    with c1:
        fig = px.pie(
            values=attrition_counts.values,
            names=attrition_counts.index,
            color_discrete_sequence=["#0d9488", "#ef4444"],
            title="Attrition Distribution"
        )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("**Class Counts:**")
        for label, count in attrition_counts.items():
            pct = count / info['total_rows'] * 100
            st.metric(label, f"{count:,} ({pct:.1f}%)")
        st.info(f"⚠️ **Class imbalance present** — Attrition rate is {info['attrition_rate']}%. Models handle this with class_weight='balanced'.")

with tab3:
    # Numeric feature distributions
    numeric_cols = info["numeric_cols"]
    selected_num = st.selectbox("Select numeric feature", numeric_cols)
    if selected_num:
        fig_hist = px.histogram(
            df, x=selected_num, color="Attrition",
            barmode="overlay",
            color_discrete_sequence=["#0d9488", "#ef4444"],
            title=f"{selected_num} distribution by Attrition"
        )
        fig_hist.update_layout(paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_hist, use_container_width=True)

    cat_cols = info["categorical_cols"]
    cat_cols_filtered = [c for c in cat_cols if c != "Attrition"]
    if cat_cols_filtered:
        selected_cat = st.selectbox("Select categorical feature", cat_cols_filtered)
        vc = df[selected_cat].value_counts().reset_index()
        vc.columns = [selected_cat, "Count"]
        fig_bar = px.bar(
            vc, x=selected_cat, y="Count",
            color="Count",
            color_continuous_scale=["#0891b2", "#0d9488"],
            title=f"{selected_cat} value counts"
        )
        fig_bar.update_layout(paper_bgcolor="rgba(0,0,0,0)", coloraxis_showscale=False)
        st.plotly_chart(fig_bar, use_container_width=True)

with tab4:
    if "Department" in df.columns:
        dept_attrition = df.groupby(["Department", "Attrition"]).size().reset_index(name="Count")
        fig_dept = px.bar(
            dept_attrition, x="Department", y="Count", color="Attrition",
            barmode="group",
            color_discrete_sequence=["#0d9488", "#ef4444"],
            title="Employee Counts by Department and Attrition Status"
        )
        fig_dept.update_layout(paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_dept, use_container_width=True)

    if "JobRole" in df.columns:
        role_attrition = df.groupby(["JobRole", "Attrition"]).size().reset_index(name="Count")
        fig_role = px.bar(
            role_attrition, x="JobRole", y="Count", color="Attrition",
            barmode="group",
            color_discrete_sequence=["#0d9488", "#ef4444"],
            title="Employee Counts by Job Role and Attrition Status"
        )
        fig_role.update_layout(paper_bgcolor="rgba(0,0,0,0)", xaxis_tickangle=-35)
        st.plotly_chart(fig_role, use_container_width=True)

    if "MonthlyIncome" in df.columns:
        fig_box = px.box(
            df, x="Department", y="MonthlyIncome", color="Attrition",
            color_discrete_sequence=["#0d9488", "#ef4444"],
            title="Monthly Income Distribution by Department and Attrition"
        )
        fig_box.update_layout(paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_box, use_container_width=True)
