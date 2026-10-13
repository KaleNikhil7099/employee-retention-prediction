"""
auth/auth_pages.py — Login and registration UI for RetentionAI.
Professional design: navy/teal theme with centered card.
"""

import streamlit as st
from auth.db import init_db, register_user, authenticate_user


# ── Shared CSS ────────────────────────────────────────────────────────────────
AUTH_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
    min-height: 100vh;
}

.auth-container {
    max-width: 440px;
    margin: 60px auto 0;
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(99, 211, 196, 0.2);
    border-radius: 20px;
    padding: 40px 36px;
    backdrop-filter: blur(20px);
    box-shadow: 0 25px 50px rgba(0,0,0,0.4);
}

.auth-logo {
    text-align: center;
    margin-bottom: 8px;
}

.auth-logo-icon {
    font-size: 42px;
    display: block;
    margin-bottom: 4px;
}

.auth-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 28px;
    font-weight: 700;
    color: #f8fafc;
    text-align: center;
    margin: 0;
}

.auth-subtitle {
    font-size: 13px;
    color: #94a3b8;
    text-align: center;
    margin: 6px 0 28px;
}

.auth-divider {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.08);
    margin: 20px 0;
}

.teal-badge {
    display: inline-block;
    background: linear-gradient(90deg, #0d9488, #0891b2);
    color: white;
    font-size: 11px;
    font-weight: 600;
    padding: 2px 10px;
    border-radius: 20px;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    margin-bottom: 16px;
}

/* Override Streamlit input styles */
input[type="text"], input[type="email"], input[type="password"] {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 10px !important;
    color: #f1f5f9 !important;
}

.stTextInput label, .stSelectbox label {
    color: #cbd5e1 !important;
    font-weight: 500;
    font-size: 14px;
}

div[data-testid="stAlert"] {
    border-radius: 10px !important;
}
</style>
"""


def _inject_css():
    st.markdown(AUTH_CSS, unsafe_allow_html=True)


def show_login_page():
    """Render the professional login page."""
    init_db()
    _inject_css()

    # Center card
    _, col, _ = st.columns([1, 2, 1])
    with col:
        st.markdown("""
        <div class="auth-container">
            <div class="auth-logo">
                <span class="auth-logo-icon">🧬</span>
            </div>
            <p class="auth-title">RetentionAI</p>
            <p class="auth-subtitle">Intelligent Employee Retention Platform</p>
            <hr class="auth-divider">
            <div style="text-align:center">
                <span class="teal-badge">🔒 Secure Login</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        with st.form("login_form"):
            email = st.text_input("📧 Email Address", placeholder="you@company.com")
            password = st.text_input("🔑 Password", type="password", placeholder="Enter your password")
            col_btn, col_sw = st.columns([2, 1])
            with col_btn:
                submitted = st.form_submit_button("Sign In →", use_container_width=True, type="primary")
            with col_sw:
                switch = st.form_submit_button("Register", use_container_width=True)

        if switch:
            st.session_state.auth_mode = "register"
            st.rerun()

        if submitted:
            if not email or not password:
                st.error("Please fill in both fields.")
            else:
                ok, result = authenticate_user(email, password)
                if ok:
                    st.session_state.authenticated = True
                    st.session_state.user = result
                    st.session_state.auth_mode = "logged_in"
                    st.success(f"Welcome back, **{result['username']}**! 🎉")
                    st.rerun()
                else:
                    st.error(f"❌ {result}")

        st.caption("Don't have an account? Click **Register** above.")


def show_register_page():
    """Render the registration page."""
    init_db()
    _inject_css()

    _, col, _ = st.columns([1, 2, 1])
    with col:
        st.markdown("""
        <div class="auth-container">
            <div class="auth-logo">
                <span class="auth-logo-icon">🧬</span>
            </div>
            <p class="auth-title">RetentionAI</p>
            <p class="auth-subtitle">Create your account</p>
            <hr class="auth-divider">
            <div style="text-align:center">
                <span class="teal-badge">✨ New Account</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        with st.form("register_form"):
            username = st.text_input("👤 Username", placeholder="johndoe")
            email = st.text_input("📧 Email Address", placeholder="you@company.com")
            password = st.text_input("🔑 Password", type="password", placeholder="Min. 8 characters")
            confirm = st.text_input("🔑 Confirm Password", type="password", placeholder="Re-enter password")
            col_btn, col_sw = st.columns([2, 1])
            with col_btn:
                submitted = st.form_submit_button("Create Account →", use_container_width=True, type="primary")
            with col_sw:
                switch = st.form_submit_button("Back to Login", use_container_width=True)

        if switch:
            st.session_state.auth_mode = "login"
            st.rerun()

        if submitted:
            if not all([username, email, password, confirm]):
                st.error("All fields are required.")
            elif password != confirm:
                st.error("Passwords do not match.")
            else:
                ok, msg = register_user(username, email, password)
                if ok:
                    st.success("✅ Account created! Please sign in.")
                    st.session_state.auth_mode = "login"
                    st.rerun()
                else:
                    st.error(f"❌ {msg}")

        st.caption("Already have an account? Click **Back to Login** above.")


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
