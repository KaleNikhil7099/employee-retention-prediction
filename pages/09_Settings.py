"""
pages/09_Settings.py — User settings and logout.
"""

import streamlit as st
from auth.auth_pages import logout
from auth.db import get_user_by_email
from ml.train_pipeline import load_metrics, PIPELINE_PATH
from ml.data_loader import dataset_exists
import os

st.markdown("# ⚙️ Settings & Account")

user = st.session_state.get("user", {})

# ── Account info ──────────────────────────────────────────────────────────────
st.markdown("### 👤 Account Information")
col1, col2 = st.columns(2)
col1.markdown(f"**Username:** {user.get('username', 'N/A')}")
col1.markdown(f"**Email:** {user.get('email', 'N/A')}")
col2.markdown(f"**Account Created:** {user.get('created_at', 'N/A')[:10]}")

st.divider()

# ── System status ─────────────────────────────────────────────────────────────
st.markdown("### 🔧 System Status")
col1, col2, col3 = st.columns(3)

with col1:
    if dataset_exists():
        st.success("✅ Dataset: Found")
    else:
        st.error("❌ Dataset: Not found")

with col2:
    if os.path.isfile(PIPELINE_PATH):
        st.success("✅ Model: Trained")
    else:
        st.warning("⏳ Model: Not trained")

with col3:
    hf_token = os.environ.get("HF_API_TOKEN", "")
    if hf_token:
        st.success("✅ HF Token: Set")
    else:
        st.warning("⚠️ HF Token: Not set")

st.divider()

# ── App info ──────────────────────────────────────────────────────────────────
st.markdown("### ℹ️ Application Information")
metrics = load_metrics()
if metrics:
    st.markdown(f"**Best Model:** {metrics.get('best_model', 'N/A')}")
    bm = metrics["models"].get(metrics.get("best_model", ""), {})
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{bm.get('accuracy', 'N/A')}")
    col2.metric("F1-Score", f"{bm.get('f1_score', 'N/A')}")
    col3.metric("ROC-AUC", f"{bm.get('roc_auc', 'N/A')}")

st.markdown("""
| Detail | Value |
|---|---|
| App Version | 1.0.0 |
| Dataset | IBM HR Analytics (Fictional) |
| Framework | Streamlit |
| Models | Logistic Regression, Random Forest, Gradient Boosting |
| Python | 3.11+ |
""")

st.divider()

# ── Logout ─────────────────────────────────────────────────────────────────────
st.markdown("### 🚪 Sign Out")
st.caption("Logging out will clear your session data. You will need to log in again.")

if st.button("🚪 Logout", type="primary"):
    logout()
