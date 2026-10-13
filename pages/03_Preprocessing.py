"""
pages/03_Preprocessing.py — Data Preprocessing Summary
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from ml.data_loader import load_raw_dataset, prepare_features, DROP_COLS, dataset_exists

st.markdown("# ⚙️ Data Preprocessing")
st.caption("Summary of the preprocessing pipeline applied before model training.")

if not dataset_exists():
    st.error("Dataset not found. Please add the IBM HR CSV to the `data/` folder.")
    st.stop()

@st.cache_data
def compute_preprocessing_summary():
    df = load_raw_dataset()
    X, y, categorical_cols, numeric_cols = prepare_features(df)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return df, X, y, X_train, X_test, y_train, y_test, categorical_cols, numeric_cols

df, X, y, X_train, X_test, y_train, y_test, cat_cols, num_cols = compute_preprocessing_summary()
st.session_state.preprocessing_done = True

# ── Summary cards ─────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("📊 Total Features", len(X.columns))
col2.metric("🔢 Numeric Features", len(num_cols))
col3.metric("🔤 Categorical Features", len(cat_cols))
col4.metric("🧹 Dropped Columns", len([c for c in DROP_COLS if c in df.columns]))

st.divider()

tab1, tab2, tab3, tab4 = st.tabs(["🔧 Pipeline Steps", "🔢 Numeric Cols", "🔤 Categorical Cols", "✂️ Train/Test Split"])

with tab1:
    st.markdown("#### Preprocessing Pipeline Steps")
    steps = [
        ("1. Drop Constant/ID Columns", f"Removed: `{'`, `'.join([c for c in DROP_COLS if c in df.columns])}`", "✅"),
        ("2. Separate Features & Target", f"Target: `Attrition` → binary (Yes=1, No=0)", "✅"),
        ("3. StandardScaler", f"Applied to {len(num_cols)} numeric columns", "✅"),
        ("4. OrdinalEncoder", f"Applied to {len(cat_cols)} categorical columns (unknown → -1)", "✅"),
        ("5. Stratified Train/Test Split", f"80% train ({len(X_train):,} rows) / 20% test ({len(X_test):,} rows)", "✅"),
        ("6. Fit Only on Train Data", "Transformers fitted on X_train only — no data leakage", "✅"),
    ]
    for title, detail, status in steps:
        with st.expander(f"{status} {title}"):
            st.markdown(detail)

with tab2:
    st.markdown(f"#### {len(num_cols)} Numeric Features")
    stats = X[num_cols].describe().T.round(2)
    st.dataframe(stats, use_container_width=True)

with tab3:
    st.markdown(f"#### {len(cat_cols)} Categorical Features")
    cat_info = pd.DataFrame({
        "Column": cat_cols,
        "Unique Values": [X[c].nunique() for c in cat_cols],
        "Values": [", ".join(map(str, X[c].unique()[:8])) for c in cat_cols]
    })
    st.dataframe(cat_info, use_container_width=True)

with tab4:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Training Set**")
        st.metric("Samples", f"{len(X_train):,}")
        st.metric("Attrition Rate", f"{y_train.mean()*100:.2f}%")
        fig_train = px.pie(
            values=[int((y_train == 0).sum()), int((y_train == 1).sum())],
            names=["Stay", "Leave"],
            color_discrete_sequence=["#0d9488", "#ef4444"],
            title="Train Split"
        )
        fig_train.update_layout(paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_train, use_container_width=True)
    with c2:
        st.markdown("**Test Set**")
        st.metric("Samples", f"{len(X_test):,}")
        st.metric("Attrition Rate", f"{y_test.mean()*100:.2f}%")
        fig_test = px.pie(
            values=[int((y_test == 0).sum()), int((y_test == 1).sum())],
            names=["Stay", "Leave"],
            color_discrete_sequence=["#0d9488", "#ef4444"],
            title="Test Split"
        )
        fig_test.update_layout(paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_test, use_container_width=True)
