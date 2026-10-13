"""
app.py — RetentionAI main entry point.
"""

import streamlit as st
from auth.auth_pages import check_auth, inject_global_css
from auth.db import init_db
import importlib.util, os as _os

st.set_page_config(
    page_title="RetentionAI — HR Analytics Platform",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_db()

if not check_auth():
    st.stop()

# ── Inject global CSS for authenticated pages ─────────────────────────────────
inject_global_css()

user = st.session_state.get("user", {})

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="padding:20px 16px 8px; border-bottom:1px solid rgba(255,255,255,0.08); margin-bottom:16px;">
        <div style="font-family:'Space Grotesk',sans-serif; font-size:22px; font-weight:700; color:#f8fafc; letter-spacing:-0.5px;">
            🧬 RetentionAI
        </div>
        <div style="font-size:11px; color:#475569; margin-top:2px; letter-spacing:0.3px;">HR ANALYTICS PLATFORM</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background:rgba(13,148,136,0.12); border:1px solid rgba(13,148,136,0.25); border-radius:10px; padding:10px 14px; margin:0 0 16px;">
        <div style="font-size:12px; font-weight:600; color:#5eead4;">👤 {user.get('username','User')}</div>
        <div style="font-size:11px; color:#475569; margin-top:2px;">{user.get('email','')}</div>
    </div>
    """, unsafe_allow_html=True)

    # Navigation pages and their icons
    NAV_ITEMS = [
        ("🏠", "Dashboard",         "pages/01_Dashboard.py"),
        ("📦", "Dataset Explorer",  "pages/02_Dataset_Explorer.py"),
        ("⚙️", "Preprocessing",     "pages/03_Preprocessing.py"),
        ("🧪", "Model Training",    "pages/04_Model_Training.py"),
        ("🔮", "Single Prediction", "pages/05_Prediction.py"),
        ("📁", "Batch Prediction",  "pages/06_Batch_Prediction.py"),
        ("🤖", "AI Chatbot",        "pages/07_AI_Chatbot.py"),
        ("📊", "Model Insights",    "pages/08_Model_Insights.py"),
        ("⚙️", "Settings",          "pages/09_Settings.py"),
    ]

    if "current_page" not in st.session_state:
        st.session_state.current_page = "Dashboard"

    st.markdown("<div style='font-size:11px; font-weight:700; color:#475569; letter-spacing:1px; padding:0 4px 8px;'>NAVIGATION</div>", unsafe_allow_html=True)

    for icon, label, path in NAV_ITEMS:
        is_active = st.session_state.current_page == label
        btn_style = "background:rgba(13,148,136,0.15) !important; color:#5eead4 !important; border-left:3px solid #0d9488 !important;" if is_active else ""
        if st.button(f"{icon}  {label}", key=f"nav_{label}", use_container_width=True):
            st.session_state.current_page = label
            st.rerun()

    st.markdown("<div style='margin-top:auto;'></div>", unsafe_allow_html=True)
    st.divider()

    # Workflow status
    dataset_ok = st.session_state.get("dataset_loaded", False)
    model_ok = st.session_state.get("model_trained", False)
    pred_ok = st.session_state.get("prediction_made", False)

    st.markdown("""<div style='font-size:11px; font-weight:700; color:#475569; letter-spacing:1px; padding:0 4px 8px;'>WORKFLOW STATUS</div>""", unsafe_allow_html=True)
    
    def _status_row(label, done):
        icon = "✅" if done else "⏳"
        color = "#5eead4" if done else "#475569"
        return f'<div style="font-size:12px; color:{color}; padding:3px 4px;">{icon} {label}</div>'

    st.markdown(
        _status_row("Dataset loaded", dataset_ok) +
        _status_row("Model trained", model_ok) +
        _status_row("Prediction made", pred_ok),
        unsafe_allow_html=True
    )

# ── Page routing ──────────────────────────────────────────────────────────────
PAGE_MAP = {
    "Dashboard":         "pages/01_Dashboard.py",
    "Dataset Explorer":  "pages/02_Dataset_Explorer.py",
    "Preprocessing":     "pages/03_Preprocessing.py",
    "Model Training":    "pages/04_Model_Training.py",
    "Single Prediction": "pages/05_Prediction.py",
    "Batch Prediction":  "pages/06_Batch_Prediction.py",
    "AI Chatbot":        "pages/07_AI_Chatbot.py",
    "Model Insights":    "pages/08_Model_Insights.py",
    "Settings":          "pages/09_Settings.py",
}

def _run_page(rel_path: str):
    abs_path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), rel_path)
    spec = importlib.util.spec_from_file_location("_page", abs_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

current = st.session_state.get("current_page", "Dashboard")
if current in PAGE_MAP:
    _run_page(PAGE_MAP[current])
