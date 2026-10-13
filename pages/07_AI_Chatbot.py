"""
pages/07_AI_Chatbot.py — RetentionAI AI Chatbot with HF Inference API.
Grounds responses in real dataset statistics and model metrics.
"""

import streamlit as st
import os
import json
import requests
from ml.data_loader import load_raw_dataset, get_dataset_info, get_dept_attrition, get_role_attrition, dataset_exists
from ml.train_pipeline import load_metrics, PIPELINE_PATH

st.markdown("# 🤖 RetentionAI Assistant")
st.caption("Ask questions about the dataset, model, and employee retention strategies.")

HF_API_TOKEN = os.environ.get("HF_API_TOKEN", "")
HF_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"

if not HF_API_TOKEN:
    st.warning("""
    **🔑 Hugging Face API Token Required**
    
    Set the environment variable `HF_API_TOKEN=hf_...` before running.
    
    Get your free token at: https://huggingface.co/settings/tokens
    
    The dataset analytics features below are still available without the API token.
    """)


# ── Build context from real data ───────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def build_context() -> str:
    """Build a grounded context string from actual dataset and model metrics."""
    sections = []

    if dataset_exists():
        try:
            df = load_raw_dataset()
            info = get_dataset_info(df)
            dept_df = get_dept_attrition(df)
            role_df = get_role_attrition(df)

            sections.append(f"""=== DATASET FACTS (IBM HR Analytics) ===
- Total employees: {info['total_rows']}
- Total features: {info['total_cols']}
- Employees who left (Attrition=Yes): {info['attrition_yes']}
- Employees who stayed (Attrition=No): {info['attrition_no']}
- Overall attrition rate: {info['attrition_rate']}%
- Missing values: {info['missing_values']}
- Duplicate rows: {info['duplicate_rows']}
- Note: This is a FICTIONAL IBM HR dataset for analytics purposes, not real employee records.""")

            if not dept_df.empty:
                top_dept = dept_df.iloc[0]
                sections.append(f"""
=== DEPARTMENT ANALYSIS ===
- Highest attrition dept: {top_dept['Department']} ({top_dept['Attrition Rate (%)']:.1f}%)
- All departments: {dept_df.to_string(index=False)}""")

            if not role_df.empty:
                top_role = role_df.iloc[0]
                sections.append(f"""
=== JOB ROLE ANALYSIS ===
- Highest attrition role: {top_role['Job Role']} ({top_role['Attrition Rate (%)']:.1f}%)""")

        except Exception as e:
            sections.append(f"Dataset unavailable: {e}")

    metrics = load_metrics()
    if metrics:
        best = metrics.get("best_model", "N/A")
        bm = metrics["models"].get(best, {})
        sections.append(f"""
=== MODEL PERFORMANCE ===
- Best model: {best}
- Accuracy: {bm.get('accuracy', 'N/A')}
- F1-Score: {bm.get('f1_score', 'N/A')}
- Precision: {bm.get('precision', 'N/A')}
- Recall: {bm.get('recall', 'N/A')}
- ROC-AUC: {bm.get('roc_auc', 'N/A')}
- Models compared: {', '.join(metrics['models'].keys())}""")

    # Last prediction context
    if st.session_state.get("last_prediction"):
        pred = st.session_state.last_prediction
        inp = st.session_state.get("last_input", {})
        sections.append(f"""
=== LAST PREDICTION ===
- Result: {pred['label']}
- Leave probability: {pred['probability_leave']}%
- Risk level: {pred['risk_level']}""")

    return "\n".join(sections) if sections else "No dataset or model data available yet."


def call_hf_api(messages_history: list, context: str) -> str:
    """Call Hugging Face Inference API with conversation history."""
    if not HF_API_TOKEN:
        return "⚠️ No HF API token configured. Set the `HF_API_TOKEN` environment variable to enable AI responses."

    system_prompt = f"""You are RetentionAI Assistant, an expert HR analytics chatbot.
You help HR professionals understand employee attrition patterns and retention strategies.

IMPORTANT RULES:
1. Use ONLY the verified data provided below. Never invent statistics.
2. Clearly say "the data shows" or "based on the dataset" when citing stats.
3. For general HR advice, clearly label it as general knowledge, not data-specific.
4. Do not access files, run code, or execute any commands.
5. Keep answers concise, professional, and actionable.

{context}

Answer the user's questions based on this data. If something is not in the data, say so clearly."""

    # Build the prompt for Mistral instruction format
    prompt_parts = [f"<s>[INST] {system_prompt}\n\nUser: {messages_history[0]['content']} [/INST]"]
    for i in range(1, len(messages_history)):
        msg = messages_history[i]
        if msg["role"] == "assistant":
            prompt_parts.append(f" {msg['content']}</s>")
        else:
            prompt_parts.append(f"[INST] {msg['content']} [/INST]")

    full_prompt = "".join(prompt_parts)

    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {
        "inputs": full_prompt,
        "parameters": {
            "max_new_tokens": 512,
            "temperature": 0.3,
            "return_full_text": False,
        }
    }

    try:
        resp = requests.post(HF_API_URL, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        result = resp.json()
        if isinstance(result, list) and result:
            text = result[0].get("generated_text", "").strip()
            return text if text else "I couldn't generate a response. Please try again."
        return str(result)
    except requests.exceptions.Timeout:
        return "⚠️ The AI service timed out. Please try again or ask a simpler question."
    except requests.exceptions.HTTPError as e:
        if "503" in str(e):
            return "⚠️ The AI model is loading (this can take 30-60s for the first request). Please wait and try again."
        return f"⚠️ AI service error: {e}"
    except Exception as e:
        return f"⚠️ Unexpected error: {e}"


# ── Suggested questions ────────────────────────────────────────────────────────
SUGGESTIONS = [
    "How many employees are in the dataset?",
    "What is the overall attrition rate?",
    "Which department has the highest attrition?",
    "What factors are most associated with employees leaving?",
    "How does the model's F1-score compare to accuracy?",
    "Suggest ways to reduce attrition in Sales.",
    "Explain the last prediction.",
]

# Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ── Controls ───────────────────────────────────────────────────────────────────
col_ctrl1, col_ctrl2 = st.columns([1, 1])
with col_ctrl1:
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

with col_ctrl2:
    st.caption(f"Model: `{HF_MODEL}`")

st.markdown("**💡 Suggested Questions:**")
cols = st.columns(3)
for i, q in enumerate(SUGGESTIONS[:6]):
    if cols[i % 3].button(q, key=f"sugg_{i}", use_container_width=True):
        st.session_state.chat_history.append({"role": "user", "content": q})
        with st.spinner("Thinking..."):
            context = build_context()
            reply = call_hf_api(st.session_state.chat_history, context)
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        st.rerun()

st.divider()

# ── Chat display ───────────────────────────────────────────────────────────────
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Input ──────────────────────────────────────────────────────────────────────
user_input = st.chat_input("Ask about the dataset, model, or employee retention...")
if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            context = build_context()
            reply = call_hf_api(st.session_state.chat_history, context)
        st.markdown(reply)

    st.session_state.chat_history.append({"role": "assistant", "content": reply})

# ── Local analytics fallback ───────────────────────────────────────────────────
st.divider()
with st.expander("📊 Quick Analytics (no AI token needed)", expanded=False):
    if dataset_exists():
        df = load_raw_dataset()
        info = get_dataset_info(df)
        st.markdown(f"""
        | Metric | Value |
        |--------|-------|
        | Total Employees | {info['total_rows']:,} |
        | Attrition Count | {info['attrition_yes']:,} |
        | Attrition Rate | {info['attrition_rate']}% |
        | Columns | {info['total_cols']} |
        """)
        metrics = load_metrics()
        if metrics:
            best = metrics.get("best_model")
            bm = metrics["models"].get(best, {})
            st.markdown(f"""
            | Model Metric | Value |
            |---|---|
            | Best Model | {best} |
            | Accuracy | {bm.get('accuracy', 'N/A')} |
            | F1-Score | {bm.get('f1_score', 'N/A')} |
            | ROC-AUC | {bm.get('roc_auc', 'N/A')} |
            """)
    else:
        st.warning("Dataset not loaded yet.")
