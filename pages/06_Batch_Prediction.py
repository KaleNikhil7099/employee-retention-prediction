"""
pages/06_Batch_Prediction.py — Batch CSV prediction.
"""

import streamlit as st
import pandas as pd
import io
import os

from ml.train_pipeline import PIPELINE_PATH
from ml.predict import get_pipeline, predict_batch, get_feature_metadata

st.markdown("# 📁 Batch Prediction")
st.caption("Upload a CSV file with employee data to generate predictions for multiple employees at once.")

if not os.path.isfile(PIPELINE_PATH):
    st.error("Model not trained yet. Please go to **Model Training** first.")
    st.stop()

@st.cache_resource(show_spinner=False)
def _get_pipeline_cached():
    return get_pipeline()

pipeline, meta = _get_pipeline_cached()
feature_info = get_feature_metadata(meta)
feature_names = feature_info["feature_names"]

# ── Instructions ──────────────────────────────────────────────────────────────
with st.expander("📋 CSV Format Requirements", expanded=False):
    st.markdown(f"""
    Your CSV must contain the following **{len(feature_names)} columns** (the `Attrition` column is optional and will be ignored):
    
    ```
    {', '.join(feature_names)}
    ```
    
    **Tips:**
    - Column names are case-sensitive.
    - You can use the IBM HR dataset format as a template.
    - The `Attrition` column will be dropped if present.
    
    [Download template CSV →]  *(Use the IBM dataset CSV as reference)*
    """)

# ── Download sample template ───────────────────────────────────────────────────
sample_df = pd.DataFrame(columns=feature_names)
csv_template = sample_df.to_csv(index=False)
st.download_button(
    "📥 Download CSV Template",
    data=csv_template,
    file_name="employee_template.csv",
    mime="text/csv"
)

st.divider()

# ── Upload ─────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader("Upload Employee CSV", type=["csv"])

if uploaded is not None:
    try:
        df_upload = pd.read_csv(uploaded)
        st.markdown(f"**Uploaded:** `{uploaded.name}` — {len(df_upload):,} rows, {len(df_upload.columns)} columns")

        # Check columns
        missing_cols = [f for f in feature_names if f not in df_upload.columns]
        extra_cols = [c for c in df_upload.columns if c not in feature_names and c != "Attrition"]

        if missing_cols:
            st.error(f"❌ Missing required columns: `{'`, `'.join(missing_cols)}`")
            st.stop()

        if extra_cols:
            st.warning(f"ℹ️ Unexpected columns (will be ignored): `{'`, `'.join(extra_cols)}`")

        if st.button("🚀 Run Batch Prediction", type="primary"):
            with st.spinner("Running predictions..."):
                result_df, _, _ = predict_batch(df_upload)

            st.success(f"✅ Predictions complete for {len(result_df):,} employees.")

            # Summary
            pred_counts = result_df["Predicted_Attrition"].value_counts()
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Processed", len(result_df))
            col2.metric("Predicted to Leave", int(pred_counts.get("Yes", 0)))
            col3.metric("Predicted to Stay", int(pred_counts.get("No", 0)))

            # Show results
            st.markdown("### Prediction Results")
            # Highlight high-risk rows
            display_cols = [c for c in result_df.columns if c in feature_names[:5]] + \
                          ["Predicted_Attrition", "Probability_Leave_%", "Risk_Level"]
            display_cols = [c for c in display_cols if c in result_df.columns]
            st.dataframe(result_df[display_cols], use_container_width=True)

            # Download
            csv_out = result_df.to_csv(index=False)
            st.download_button(
                "📥 Download Results CSV",
                data=csv_out,
                file_name="retention_predictions.csv",
                mime="text/csv",
                type="primary"
            )

            st.warning("""
            ⚠️ **Privacy Notice:** Employee data is processed locally and not stored permanently.
            Avoid uploading personally identifiable information unnecessarily.
            """)

    except Exception as e:
        st.error(f"Error processing file: {e}")
