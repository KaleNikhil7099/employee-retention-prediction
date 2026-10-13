"""
auth/auth_pages.py — Login and registration UI for RetentionAI.
Professional design: clean white card with navy/teal accents. Full text visibility.
"""

import streamlit as st
from auth.db import init_db, register_user, authenticate_user

# ── Shared CSS ────────────────────────────────────────────────────────────────
AUTH_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

/* App background */
.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0f172a 100%) !important;
    min-height: 100vh;
}

/* Hide sidebar on login/register */
section[data-testid="stSidebar"] { display: none !important; }
header[data-testid="stHeader"] { background: transparent !important; }

/* ── Input fields: ensure text is always visible ─────────────── */
input[type="text"],
input[type="email"],
input[type="password"],
input[type="number"],
textarea,
.stTextInput > div > div > input,
.stNumberInput > div > div > input {
    color: #0f172a !important;
    background-color: #ffffff !important;
    border: 2px solid #e2e8f0 !important;
    border-radius: 10px !important;
    font-size: 15px !important;
    padding: 10px 14px !important;
    caret-color: #0d9488 !important;
}

input[type="text"]:focus,
input[type="email"]:focus,
input[type="password"]:focus,
.stTextInput > div > div > input:focus {
    border-color: #0d9488 !important;
    box-shadow: 0 0 0 3px rgba(13, 148, 136, 0.15) !important;
    outline: none !important;
}

/* Labels */
.stTextInput label,
.stSelectbox label,
.stNumberInput label {
    color: #e2e8f0 !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    margin-bottom: 4px !important;
}

/* Primary button styling */
.stButton > button[kind="primary"],
button[data-testid="baseButton-primary"] {
    background: linear-gradient(135deg, #0d9488, #0891b2) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    padding: 12px 20px !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
    cursor: pointer !important;
}

.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #0f766e, #0369a1) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(13, 148, 136, 0.4) !important;
}

/* Secondary button */
.stButton > button[kind="secondary"],
button[data-testid="baseButton-secondary"] {
    background: rgba(255,255,255,0.1) !important;
    color: #e2e8f0 !important;
    border: 1.5px solid rgba(255,255,255,0.2) !important;
    border-radius: 10px !important;
    font-weight: 500 !important;
    width: 100% !important;
    cursor: pointer !important;
}

.stButton > button[kind="secondary"]:hover {
    background: rgba(255,255,255,0.2) !important;
    border-color: rgba(255,255,255,0.4) !important;
}

/* Alert/error boxes */
div[data-testid="stAlert"] {
    border-radius: 10px !important;
    font-weight: 500 !important;
}

/* Caption text */
.stApp p.caption, .stApp small, .stCaption {
    color: #94a3b8 !important;
}
</style>
"""

GLOBAL_APP_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
h1, h2, h3, h4 { font-family: 'Space Grotesk', sans-serif !important; color: #0f172a !important; }

/* Main background - clean light */
.stApp { background: #f1f5f9 !important; }

/* ── All input fields: dark text on white bg ── */
input[type="text"],
input[type="email"],
input[type="password"],
input[type="number"],
textarea,
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stSelectbox > div > div > div {
    color: #0f172a !important;
    background-color: #ffffff !important;
    border: 2px solid #e2e8f0 !important;
    border-radius: 10px !important;
    font-size: 14px !important;
    caret-color: #0d9488 !important;
}

input:focus, textarea:focus {
    border-color: #0d9488 !important;
    box-shadow: 0 0 0 3px rgba(13,148,136,0.15) !important;
    outline: none !important;
}

/* Labels */
.stTextInput label, .stSelectbox label, .stNumberInput label,
.stSlider label, .stFileUploader label, [data-testid="stWidgetLabel"] {
    color: #1e293b !important;
    font-weight: 600 !important;
    font-size: 13px !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0f172a !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
    padding-top: 0 !important;
}
section[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] strong { color: #f8fafc !important; }
section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.1) !important; }

/* Sidebar nav buttons */
section[data-testid="stSidebar"] .stButton > button {
    background: transparent !important;
    color: #cbd5e1 !important;
    border: none !important;
    border-radius: 8px !important;
    text-align: left !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 8px 12px !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: all 0.15s ease !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(255,255,255,0.08) !important;
    color: #f8fafc !important;
}

/* Main content primary button */
.stButton > button[kind="primary"],
button[data-testid="baseButton-primary"] {
    background: linear-gradient(135deg, #0d9488, #0891b2) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    padding: 10px 20px !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
}
.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #0f766e, #0369a1) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 15px rgba(13,148,136,0.35) !important;
}

/* Secondary button */
.stButton > button:not([kind="primary"]) {
    background: white !important;
    color: #0f172a !important;
    border: 2px solid #e2e8f0 !important;
    border-radius: 10px !important;
    font-weight: 500 !important;
    cursor: pointer !important;
}
.stButton > button:not([kind="primary"]):hover {
    border-color: #0d9488 !important;
    color: #0d9488 !important;
}

/* Metric cards */
div[data-testid="metric-container"] {
    background: white !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 14px !important;
    padding: 16px 20px !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06) !important;
}
div[data-testid="metric-container"] label {
    color: #64748b !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #0f172a !important;
    font-size: 26px !important;
    font-weight: 700 !important;
}

/* Dataframe */
.stDataFrame { border-radius: 10px !important; overflow: hidden !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: white !important;
    border-radius: 10px !important;
    padding: 4px !important;
    gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    color: #64748b !important;
    font-weight: 500 !important;
    border-radius: 8px !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #0d9488, #0891b2) !important;
    color: white !important;
}

/* Alerts */
div[data-testid="stAlert"] { border-radius: 10px !important; }

/* Selectbox dropdown */
[data-baseweb="select"] div {
    color: #0f172a !important;
    background: white !important;
}

/* Progress bar */
.stProgress > div > div {
    background: linear-gradient(90deg, #0d9488, #0891b2) !important;
    border-radius: 6px !important;
}

/* Chat input */
.stChatInput textarea {
    color: #0f172a !important;
    background: white !important;
    border: 2px solid #e2e8f0 !important;
    border-radius: 10px !important;
}

/* Expanders */
.streamlit-expanderHeader {
    background: white !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    color: #0f172a !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: white !important;
    border-radius: 10px !important;
    border: 2px dashed #e2e8f0 !important;
}

/* Slider */
.stSlider [data-testid="stThumbValue"] { color: #0f172a !important; }
</style>
"""


def _inject_auth_css():
    st.markdown(AUTH_CSS, unsafe_allow_html=True)


def show_login_page():
    """Render the professional login page."""
    init_db()
    _inject_auth_css()

    _, col, _ = st.columns([1, 1.5, 1])
    with col:
        st.markdown("""
        <div style="text-align:center; padding: 32px 0 8px;">
            <div style="font-size:52px; margin-bottom:8px;">🧬</div>
            <h1 style="font-family:'Space Grotesk',sans-serif; font-size:32px; font-weight:700; color:#f8fafc; margin:0;">RetentionAI</h1>
            <p style="color:#94a3b8; font-size:14px; margin-top:6px;">Intelligent Employee Retention Platform</p>
        </div>
        <div style="background:rgba(255,255,255,0.07); border:1px solid rgba(255,255,255,0.12); border-radius:20px; padding:36px; backdrop-filter:blur(20px); margin-top:8px;">
            <div style="text-align:center; margin-bottom:24px;">
                <span style="background:linear-gradient(90deg,#0d9488,#0891b2); color:white; font-size:11px; font-weight:700; padding:4px 14px; border-radius:20px; letter-spacing:1px; text-transform:uppercase;">🔒 Secure Login</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        with st.form("login_form", clear_on_submit=False):
            email = st.text_input("📧 Email Address", placeholder="you@company.com", key="login_email")
            password = st.text_input("🔑 Password", type="password", placeholder="Enter your password", key="login_pass")
            
            col_btn, col_sw = st.columns([3, 2])
            with col_btn:
                submitted = st.form_submit_button("Sign In →", use_container_width=True, type="primary")
            with col_sw:
                switch = st.form_submit_button("Register →", use_container_width=True)

        if switch:
            st.session_state.auth_mode = "register"
            st.rerun()

        if submitted:
            if not email or not password:
                st.error("⚠️ Please fill in both email and password.")
            else:
                ok, result = authenticate_user(email, password)
                if ok:
                    st.session_state.authenticated = True
                    st.session_state.user = result
                    st.session_state.auth_mode = "logged_in"
                    st.success(f"✅ Welcome back, **{result['username']}**!")
                    st.rerun()
                else:
                    st.error(f"❌ {result}")

        st.markdown("""
        <p style="text-align:center; color:#64748b; font-size:13px; margin-top:16px;">
            Don't have an account? Click <strong style="color:#0d9488;">Register →</strong> above.
        </p>
        """, unsafe_allow_html=True)


def show_register_page():
    """Render the registration page."""
    init_db()
    _inject_auth_css()

    _, col, _ = st.columns([1, 1.5, 1])
    with col:
        st.markdown("""
        <div style="text-align:center; padding: 32px 0 8px;">
            <div style="font-size:52px; margin-bottom:8px;">🧬</div>
            <h1 style="font-family:'Space Grotesk',sans-serif; font-size:32px; font-weight:700; color:#f8fafc; margin:0;">RetentionAI</h1>
            <p style="color:#94a3b8; font-size:14px; margin-top:6px;">Create your free account</p>
        </div>
        <div style="background:rgba(255,255,255,0.07); border:1px solid rgba(255,255,255,0.12); border-radius:20px; padding:36px; backdrop-filter:blur(20px); margin-top:8px;">
            <div style="text-align:center; margin-bottom:24px;">
                <span style="background:linear-gradient(90deg,#7c3aed,#0891b2); color:white; font-size:11px; font-weight:700; padding:4px 14px; border-radius:20px; letter-spacing:1px; text-transform:uppercase;">✨ New Account</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        with st.form("register_form", clear_on_submit=False):
            username = st.text_input("👤 Username", placeholder="johndoe", key="reg_username")
            email = st.text_input("📧 Email Address", placeholder="you@company.com", key="reg_email")
            password = st.text_input("🔑 Password", type="password", placeholder="Min. 8 characters", key="reg_pass")
            confirm = st.text_input("🔑 Confirm Password", type="password", placeholder="Re-enter password", key="reg_confirm")

            col_btn, col_sw = st.columns([3, 2])
            with col_btn:
                submitted = st.form_submit_button("Create Account →", use_container_width=True, type="primary")
            with col_sw:
                switch = st.form_submit_button("← Login", use_container_width=True)

        if switch:
            st.session_state.auth_mode = "login"
            st.rerun()

        if submitted:
            if not all([username, email, password, confirm]):
                st.error("⚠️ All fields are required.")
            elif password != confirm:
                st.error("❌ Passwords do not match.")
            elif len(password) < 8:
                st.error("❌ Password must be at least 8 characters.")
            else:
                ok, msg = register_user(username, email, password)
                if ok:
                    st.success("✅ Account created! Redirecting to login...")
                    st.session_state.auth_mode = "login"
                    st.rerun()
                else:
                    st.error(f"❌ {msg}")

        st.markdown("""
        <p style="text-align:center; color:#64748b; font-size:13px; margin-top:16px;">
            Already have an account? Click <strong style="color:#0d9488;">← Login</strong> above.
        </p>
        """, unsafe_allow_html=True)


def check_auth():
    """
    Gate function — call at the top of every page.
    Returns True if authenticated; otherwise renders auth UI and returns False.
    """
    init_db()
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "auth_mode" not in st.session_state:
        st.session_state.auth_mode = "login"

    if st.session_state.authenticated:
        return True

    if st.session_state.auth_mode == "register":
        show_register_page()
    else:
        show_login_page()
    return False


def logout():
    """Clear all sensitive session data."""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()


def inject_global_css():
    """Inject global CSS for authenticated pages."""
    st.markdown(GLOBAL_APP_CSS, unsafe_allow_html=True)
