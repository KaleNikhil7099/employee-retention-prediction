"""
app.py — RetentionAI main entry point.
Routes between auth pages and the main multi-page application.
"""

import streamlit as st
from auth.auth_pages import check_auth
from auth.db import init_db

# Page config must be first
st.set_page_config(
    page_title="RetentionAI — HR Analytics Platform",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Global CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; }
.stApp { background: #f0f4f8; }

.kpi-card {
    background: white;
    border-radius: 14px;
    padding: 20px 24px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}

.nav-header {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 18px;
    font-weight: 700;
    color: #0f172a;
    padding: 4px 0 12px;
}

.status-dot-green { color: #10b981; }
.status-dot-orange { color: #f59e0b; }
.status-dot-gray { color: #94a3b8; }

section[data-testid="stSidebar"] {
    background: #0f172a !important;
    border-right: 1px solid rgba(255,255,255,0.06);
}

section[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 { color: #f8fafc !important; }

div[data-testid="stAlert"] { border-radius: 10px !important; }
</style>
""", unsafe_allow_html=True)

init_db()

if not check_auth():
    st.stop()

# ── Authenticated area ───────────────────────────────────────────────────────
user = st.session_state.get("user", {})

with st.sidebar:
    st.markdown(f"""
    <div style="padding:16px 0 8px">
        <div style="font-family:'Space Grotesk',sans-serif;font-size:22px;font-weight:700;color:#f8fafc">
            🧬 RetentionAI
        </div>
        <div style="font-size:12px;color:#64748b;margin-top:2px">HR Analytics Platform</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background:rgba(255,255,255,0.05);border-radius:10px;padding:12px 14px;margin:8px 0 16px">
        <div style="font-size:13px;font-weight:600;color:#e2e8f0">👤 {user.get('username','User')}</div>
        <div style="font-size:11px;color:#64748b">{user.get('email','')}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Navigate**")
    pages = {
        "🏠 Dashboard": "pages/01_Dashboard.py",
        "📦 Dataset Explorer": "pages/02_Dataset_Explorer.py",
        "⚙️ Preprocessing": "pages/03_Preprocessing.py",
        "🧪 Model Training": "pages/04_Model_Training.py",
        "🔮 Single Prediction": "pages/05_Prediction.py",
        "📁 Batch Prediction": "pages/06_Batch_Prediction.py",
        "🤖 AI Chatbot": "pages/07_AI_Chatbot.py",
        "📊 Model Insights": "pages/08_Model_Insights.py",
        "⚙️ Settings": "pages/09_Settings.py",
    }

    if "current_page" not in st.session_state:
        st.session_state.current_page = "🏠 Dashboard"

    for label in pages:
        if st.button(label, key=f"nav_{label}", use_container_width=True):
            st.session_state.current_page = label
            st.rerun()

    st.divider()

    # Workflow status
    st.markdown("**Workflow Status**")
    dataset_ok = st.session_state.get("dataset_loaded", False)
    model_ok = st.session_state.get("model_trained", False)
    st.caption(f"{'✅' if dataset_ok else '⏳'} Dataset loaded")
    st.caption(f"{'✅' if model_ok else '⏳'} Model trained")

# ── Route to page ────────────────────────────────────────────────────────────
import importlib.util, os as _os

def _run_page(rel_path: str):
    """Load and exec a page module in its own namespace with project root on sys.path."""
    abs_path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), rel_path)
    spec = importlib.util.spec_from_file_location("_page", abs_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

current = st.session_state.get("current_page", "🏠 Dashboard")

PAGE_MAP = {
    "🏠 Dashboard":          "pages/01_Dashboard.py",
    "📦 Dataset Explorer":   "pages/02_Dataset_Explorer.py",
    "⚙️ Preprocessing":      "pages/03_Preprocessing.py",
    "🧪 Model Training":     "pages/04_Model_Training.py",
    "🔮 Single Prediction":  "pages/05_Prediction.py",
    "📁 Batch Prediction":   "pages/06_Batch_Prediction.py",
    "🤖 AI Chatbot":         "pages/07_AI_Chatbot.py",
    "📊 Model Insights":     "pages/08_Model_Insights.py",
    "⚙️ Settings":           "pages/09_Settings.py",
}

if current in PAGE_MAP:
    _run_page(PAGE_MAP[current])
