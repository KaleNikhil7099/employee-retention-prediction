"""
app.py — RetentionAI: Complete Employee Retention Analytics Platform
Clean, working, exam-ready.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import json
import importlib.util

from auth.auth_pages import check_auth, inject_global_css, logout
from auth.db import init_db

# ─── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RetentionAI — Employee Retention Platform",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_db()
if not check_auth():
    st.stop()

inject_global_css()

# ─── Constants ─────────────────────────────────────────────────────────────────
DATA_PATH   = os.path.join("data", "WA_Fn-UseC_-HR-Employee-Attrition.csv")
MODEL_PATH  = os.path.join("models", "pipeline.pkl")
METRIC_PATH = os.path.join("models", "metrics.json")
user = st.session_state.get("user", {})

# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="padding:18px 12px 12px;border-bottom:1px solid rgba(255,255,255,0.08);margin-bottom:12px">
      <div style="font-size:21px;font-weight:700;color:#f8fafc;font-family:'Space Grotesk',sans-serif">🧬 RetentionAI</div>
      <div style="font-size:11px;color:#475569;letter-spacing:.5px">HR ANALYTICS PLATFORM</div>
    </div>
    <div style="background:rgba(13,148,136,.12);border:1px solid rgba(13,148,136,.3);border-radius:9px;padding:9px 13px;margin-bottom:14px">
      <div style="font-size:12px;font-weight:600;color:#5eead4">👤 {user.get('username','User')}</div>
      <div style="font-size:11px;color:#64748b">{user.get('email','')}</div>
    </div>
    """, unsafe_allow_html=True)

    pages = [
        "🏠  Dashboard",
        "📊  Dataset Explorer",
        "⚙️  Preprocessing",
        "🧪  Model Training",
        "🔮  Predict Attrition",
        "📁  Batch Predict",
        "💡  Analytics & Insights",
        "⚙️  Settings / Logout",
    ]

    if "page" not in st.session_state:
        st.session_state.page = "🏠  Dashboard"

    for p in pages:
        active = st.session_state.page == p
        if st.button(p, key=f"nav_{p}", use_container_width=True,
                     type="primary" if active else "secondary"):
            st.session_state.page = p
            st.rerun()

    st.divider()
    has_data  = os.path.isfile(DATA_PATH)
    has_model = os.path.isfile(MODEL_PATH)
    st.markdown(f"""
    <div style="font-size:11px;color:#475569;font-weight:700;letter-spacing:.5px;margin-bottom:6px">STATUS</div>
    <div style="font-size:12px;color:{'#5eead4' if has_data  else '#ef4444'}">{'✅' if has_data  else '❌'} Dataset</div>
    <div style="font-size:12px;color:{'#5eead4' if has_model else '#ef4444'}">{'✅' if has_model else '❌'} Model trained</div>
    """, unsafe_allow_html=True)

# ─── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

@st.cache_resource(show_spinner=False)
def load_model():
    import joblib
    pkg = joblib.load(MODEL_PATH)
    return pkg["pipeline"], pkg["meta"]

def load_metrics():
    if not os.path.isfile(METRIC_PATH):
        return None
    with open(METRIC_PATH) as f:
        return json.load(f)

# ─── PAGE FUNCTIONS ────────────────────────────────────────────────────────────

# ══════════════════════════════════════════════════════════════════════════════
def page_dashboard():
    st.markdown("# 🏠 Employee Retention Dashboard")
    st.caption("Live analytics from the IBM HR Employee Attrition dataset.")

    if not os.path.isfile(DATA_PATH):
        st.error("📂 Dataset not found. Add `WA_Fn-UseC_-HR-Employee-Attrition.csv` to the `data/` folder.")
        return

    df = load_data()
    left_count  = int((df["Attrition"] == "Yes").sum())
    stay_count  = int((df["Attrition"] == "No").sum())
    total       = len(df)
    rate        = round(left_count / total * 100, 1)

    m = load_metrics()

    # ── KPI row ───────────────────────────────────────────────────────────────
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("👥 Total Employees",   f"{total:,}")
    c2.metric("🚪 Left Company",      f"{left_count:,}")
    c3.metric("✅ Still Active",       f"{stay_count:,}")
    c4.metric("📉 Attrition Rate",    f"{rate}%")
    c5.metric("🏆 Model ROC-AUC",     f"{m['models'][m['best_model']]['roc_auc']:.3f}" if m else "N/A")

    st.divider()

    # ── Attrition pie + dept bar ──────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        fig = px.pie(
            values=[stay_count, left_count], names=["Stayed","Left"],
            color_discrete_sequence=["#0d9488","#ef4444"],
            title="Overall Attrition Split", hole=0.45
        )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", margin=dict(t=40,b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        dept = df.groupby("Department")["Attrition"].apply(
            lambda s: round((s=="Yes").mean()*100,1)).reset_index()
        dept.columns = ["Department","Rate (%)"]
        fig2 = px.bar(dept, x="Department", y="Rate (%)",
                      color="Rate (%)", color_continuous_scale=["#0d9488","#f59e0b","#ef4444"],
                      title="Attrition Rate by Department", text_auto=True)
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", coloraxis_showscale=False,
                           margin=dict(t=40,b=10))
        st.plotly_chart(fig2, use_container_width=True)

    # ── Job role + overtime ───────────────────────────────────────────────────
    col3, col4 = st.columns(2)
    with col3:
        role = df.groupby("JobRole")["Attrition"].apply(
            lambda s: round((s=="Yes").mean()*100,1)).reset_index()
        role.columns = ["JobRole","Rate (%)"]
        role = role.sort_values("Rate (%)", ascending=True)
        fig3 = px.bar(role, x="Rate (%)", y="JobRole", orientation="h",
                      color="Rate (%)", color_continuous_scale=["#0d9488","#ef4444"],
                      title="Attrition Rate by Job Role", text_auto=True)
        fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", coloraxis_showscale=False,
                           margin=dict(t=40,b=10))
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        ot = df.groupby("OverTime")["Attrition"].apply(
            lambda s: round((s=="Yes").mean()*100,1)).reset_index()
        ot.columns = ["OverTime","Rate (%)"]
        fig4 = px.bar(ot, x="OverTime", y="Rate (%)",
                      color="OverTime", color_discrete_sequence=["#0d9488","#ef4444"],
                      title="Attrition Rate: Overtime vs Non-Overtime", text_auto=True)
        fig4.update_layout(paper_bgcolor="rgba(0,0,0,0)", showlegend=False,
                           margin=dict(t=40,b=10))
        st.plotly_chart(fig4, use_container_width=True)

    # ── Income analysis ───────────────────────────────────────────────────────
    st.markdown("#### 💰 Monthly Income Distribution")
    fig5 = px.histogram(df, x="MonthlyIncome", color="Attrition",
                       color_discrete_map={"Yes":"#ef4444","No":"#0d9488"},
                       barmode="overlay", nbins=40,
                       title="Monthly Income vs Attrition", opacity=0.75)
    fig5.update_layout(paper_bgcolor="rgba(0,0,0,0)", margin=dict(t=40,b=10))
    st.plotly_chart(fig5, use_container_width=True)

    # ── Model metrics ─────────────────────────────────────────────────────────
    if m:
        st.divider()
        st.markdown(f"### 🏆 Best Model: **{m['best_model']}**")
        bm = m["models"][m["best_model"]]
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Accuracy",  f"{bm['accuracy']:.1%}")
        c2.metric("Precision", f"{bm['precision']:.3f}")
        c3.metric("Recall",    f"{bm['recall']:.3f}")
        c4.metric("F1-Score",  f"{bm['f1_score']:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
def page_dataset():
    st.markdown("# 📊 Dataset Explorer")
    st.caption("IBM HR Analytics Employee Attrition & Performance — 1,470 employees, 35 features (fictional dataset).")

    if not os.path.isfile(DATA_PATH):
        st.error("Dataset not found. Please place `WA_Fn-UseC_-HR-Employee-Attrition.csv` in the `data/` folder.")
        return

    df = load_data()

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", f"{len(df.columns)}")
    c3.metric("Missing Values", f"{int(df.isnull().sum().sum())}")
    c4.metric("Duplicate Rows", f"{int(df.duplicated().sum())}")

    tab1, tab2, tab3 = st.tabs(["📋 Preview & Column Info", "📈 Feature Distributions", "🏢 Department Analysis"])

    with tab1:
        st.markdown("**First 10 rows:**")
        st.dataframe(df.head(10), use_container_width=True)

        st.markdown("**Column Summary:**")
        info = pd.DataFrame({
            "Column": df.columns,
            "Type": df.dtypes.astype(str).values,
            "Non-Null Count": df.count().values,
            "Unique Values": df.nunique().values
        })
        st.dataframe(info, hide_index=True, use_container_width=True)

    with tab2:
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = [c for c in df.select_dtypes(include="object").columns if c != "Attrition"]

        sel = st.selectbox("Select numeric feature to explore:", num_cols)
        fig = px.histogram(df, x=sel, color="Attrition",
                           color_discrete_map={"Yes":"#ef4444","No":"#0d9488"},
                           barmode="overlay", nbins=30, opacity=0.8,
                           title=f"{sel} distribution by Attrition status")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

        sel2 = st.selectbox("Select categorical feature:", cat_cols)
        cat_data = df.groupby([sel2,"Attrition"]).size().reset_index(name="Count")
        fig2 = px.bar(cat_data, x=sel2, y="Count", color="Attrition",
                      color_discrete_map={"Yes":"#ef4444","No":"#0d9488"},
                      barmode="group", title=f"{sel2} distribution by Attrition")
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", xaxis_tickangle=-30)
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.box(df, x="Department", y="MonthlyIncome", color="Attrition",
                         color_discrete_map={"Yes":"#ef4444","No":"#0d9488"},
                         title="Monthly Income by Dept & Attrition")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig2 = px.box(df, x="Department", y="YearsAtCompany", color="Attrition",
                          color_discrete_map={"Yes":"#ef4444","No":"#0d9488"},
                          title="Tenure by Dept & Attrition")
            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
def page_preprocessing():
    st.markdown("# ⚙️ Data Preprocessing Pipeline")
    if not os.path.isfile(DATA_PATH):
        st.error("Dataset not found.")
        return

    df = load_data()
    DROP  = ["EmployeeCount","EmployeeNumber","Over18","StandardHours"]
    dropped = [c for c in DROP if c in df.columns]
    X = df.drop(columns=["Attrition"] + dropped)
    y = (df["Attrition"] == "Yes").astype(int)
    cat_cols = X.select_dtypes(include="object").columns.tolist()
    num_cols = X.select_dtypes(include=np.number).columns.tolist()

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Features",   len(X.columns))
    c2.metric("Numeric Features", len(num_cols))
    c3.metric("Categorical Features", len(cat_cols))
    c4.metric("Dropped Columns", len(dropped))

    st.divider()

    st.markdown("### 📋 Preprocessing Steps")
    steps = [
        ("1️⃣  Drop Constant/ID Columns",
         f"Removed: `{'`, `'.join(dropped)}` — these are constants or identifiers that add no predictive value."),
        ("2️⃣  Encode Target Variable",
         "Attrition: `Yes → 1` (left), `No → 0` (stayed). Binary classification problem."),
        ("3️⃣  Encode Categorical Features",
         f"**OrdinalEncoder** applied to {len(cat_cols)} columns: `{'`, `'.join(cat_cols)}`"),
        ("4️⃣  Scale Numeric Features",
         f"**StandardScaler** applied to {len(num_cols)} numeric columns → zero mean, unit variance."),
        ("5️⃣  Handle Class Imbalance",
         f"Attrition rate is ~16% (imbalanced). All models use `class_weight='balanced'` to avoid biased predictions."),
        ("6️⃣  Stratified Train/Test Split",
         f"80/20 split: **{len(X_train):,} training** / **{len(X_test):,} test** records. Stratified to preserve attrition ratio."),
        ("7️⃣  No Data Leakage",
         "All transformers (scaler, encoder) are **fit only on training data**, then applied to test data."),
    ]
    for title, detail in steps:
        with st.expander(f"✅ {title}", expanded=False):
            st.markdown(detail)

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Train Set Class Distribution")
        train_dist = pd.DataFrame({"Class":["Stayed (0)","Left (1)"],
                                   "Count":[int((y_train==0).sum()), int((y_train==1).sum())]})
        fig = px.pie(train_dist, names="Class", values="Count",
                     color_discrete_sequence=["#0d9488","#ef4444"], hole=0.4)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
        st.metric("Train Attrition Rate", f"{y_train.mean()*100:.1f}%")

    with col2:
        st.markdown("#### Test Set Class Distribution")
        test_dist = pd.DataFrame({"Class":["Stayed (0)","Left (1)"],
                                  "Count":[int((y_test==0).sum()), int((y_test==1).sum())]})
        fig2 = px.pie(test_dist, names="Class", values="Count",
                      color_discrete_sequence=["#0d9488","#ef4444"], hole=0.4)
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig2, use_container_width=True)
        st.metric("Test Attrition Rate",  f"{y_test.mean()*100:.1f}%")


# ══════════════════════════════════════════════════════════════════════════════
def page_training():
    st.markdown("# 🧪 Model Training & Evaluation")

    if not os.path.isfile(DATA_PATH):
        st.error("Dataset not found."); return

    if st.button("🚀 Train All Models (LR + RF + GBM)", type="primary", use_container_width=True):
        prog = st.progress(0)
        status = st.empty()
        def _cb(step, total, msg):
            prog.progress(step/total)
            status.info(f"⏳ {msg}")
        try:
            from ml.train_pipeline import train_all_models
            result = train_all_models(progress_callback=_cb)
            st.session_state.model_trained = True
            load_model.clear()
            st.success(f"✅ Done! Best model: **{result['best_model']}**")
            st.rerun()
        except Exception as e:
            st.error(f"Training error: {e}")
        return

    m = load_metrics()
    if not m:
        st.info("Click **Train All Models** to begin. This takes ~1 minute and trains 3 classifiers.")
        return

    st.session_state.model_trained = True
    best = m["best_model"]
    st.success(f"✅ Model trained. Best: **{best}**")

    # ── Comparison Table ──────────────────────────────────────────────────────
    st.markdown("### 📊 Model Comparison")
    rows = []
    for name, mm in m["models"].items():
        rows.append({"Model": ("🥇 " if name==best else "") + name,
                     "Accuracy":  f"{mm['accuracy']:.4f}",
                     "Precision": f"{mm['precision']:.4f}",
                     "Recall":    f"{mm['recall']:.4f}",
                     "F1-Score":  f"{mm['f1_score']:.4f}",
                     "ROC-AUC":   f"{mm['roc_auc']:.4f}"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    # ── Radar chart ───────────────────────────────────────────────────────────
    cats = ["Accuracy","Precision","Recall","F1-Score","ROC-AUC"]
    fig = go.Figure()
    colors = ["#0d9488","#f59e0b","#8b5cf6"]
    for i, (name, mm) in enumerate(m["models"].items()):
        vals = [mm["accuracy"],mm["precision"],mm["recall"],mm["f1_score"],mm["roc_auc"]]
        fig.add_trace(go.Scatterpolar(r=vals+[vals[0]], theta=cats+[cats[0]],
                                      fill="toself", name=name,
                                      line_color=colors[i], opacity=0.7))
    fig.update_layout(polar=dict(radialaxis=dict(range=[0,1])),
                      title="Model Performance Radar",
                      paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

    # ── Confusion matrix ──────────────────────────────────────────────────────
    st.markdown("### 🔢 Confusion Matrix — " + best)
    import plotly.figure_factory as ff
    cm = m["models"][best]["confusion_matrix"]
    labels = ["Stayed (0)","Left (1)"]
    fig_cm = ff.create_annotated_heatmap(z=cm, x=labels, y=labels,
                                          colorscale="Teal", showscale=True)
    fig_cm.update_layout(paper_bgcolor="rgba(0,0,0,0)", title="Confusion Matrix")
    st.plotly_chart(fig_cm, use_container_width=True)

    st.info("ℹ️ **Class Imbalance Note:** ~16% attrition rate. Recall is more critical than accuracy — we want to catch employees *at risk* of leaving.")


# ══════════════════════════════════════════════════════════════════════════════
def page_predict():
    st.markdown("# 🔮 Predict Employee Attrition")
    st.caption("Enter employee details below to predict their attrition risk.")

    if not os.path.isfile(MODEL_PATH):
        st.warning("⚠️ Model not trained yet. Go to **Model Training** first.")
        return
    if not os.path.isfile(DATA_PATH):
        st.warning("⚠️ Dataset not found.")
        return

    pipeline, meta = load_model()
    df = load_data()
    feat_names = meta["feature_names"]
    cat_cols   = meta["categorical_cols"]
    num_cols   = meta["numeric_cols"]

    # Get valid categorical values from dataset
    cat_options = {c: sorted(df[c].dropna().unique().tolist()) for c in cat_cols if c in df.columns}

    st.markdown("### 👤 Employee Information")

    # ── Organized input form ───────────────────────────────────────────────────
    with st.form("pred_form"):
        st.markdown("**Personal & Job Details**")
        r1c1, r1c2, r1c3 = st.columns(3)
        with r1c1:
            age  = st.number_input("Age", 18, 65, 35, help="Employee age in years")
            gender = st.selectbox("Gender", cat_options.get("Gender", ["Male","Female"]))
            marital = st.selectbox("Marital Status", cat_options.get("MaritalStatus", ["Single","Married","Divorced"]))
        with r1c2:
            dept = st.selectbox("Department", cat_options.get("Department", ["Sales","R&D","HR"]))
            role = st.selectbox("Job Role", cat_options.get("JobRole", []))
            edu_field = st.selectbox("Education Field", cat_options.get("EducationField", []))
        with r1c3:
            travel = st.selectbox("Business Travel", cat_options.get("BusinessTravel", ["Travel_Rarely","Travel_Frequently","Non-Travel"]))
            overtime = st.selectbox("OverTime", cat_options.get("OverTime", ["Yes","No"]))
            edu = st.slider("Education Level (1=Below College, 5=Doctor)", 1, 5, 3)

        st.divider()
        st.markdown("**Compensation & Tenure**")
        r2c1, r2c2, r2c3 = st.columns(3)
        with r2c1:
            income  = st.number_input("Monthly Income (₹)", 1009, 200000, 5000, step=500)
            daily   = st.number_input("Daily Rate", 100, 1500, 800)
            hourly  = st.number_input("Hourly Rate", 30, 100, 65)
        with r2c2:
            monthly_rate   = st.number_input("Monthly Rate", 2000, 27000, 14000)
            pct_hike       = st.slider("% Salary Hike", 11, 25, 15)
            stock          = st.slider("Stock Option Level (0-3)", 0, 3, 1)
        with r2c3:
            total_years    = st.number_input("Total Working Years", 0, 40, 10)
            num_companies  = st.slider("Num Companies Worked", 0, 9, 2)
            years_company  = st.number_input("Years at Company", 0, 40, 5)

        st.divider()
        st.markdown("**Performance & Satisfaction**")
        r3c1, r3c2, r3c3 = st.columns(3)
        with r3c1:
            job_sat  = st.slider("Job Satisfaction (1=Low, 4=High)", 1, 4, 3)
            env_sat  = st.slider("Environment Satisfaction (1-4)", 1, 4, 3)
            rel_sat  = st.slider("Relationship Satisfaction (1-4)", 1, 4, 3)
        with r3c2:
            wlb      = st.slider("Work-Life Balance (1=Bad, 4=Best)", 1, 4, 3)
            job_inv  = st.slider("Job Involvement (1-4)", 1, 4, 3)
            perf     = st.slider("Performance Rating (1-4)", 1, 4, 3)
        with r3c3:
            job_level = st.slider("Job Level (1-5)", 1, 5, 2)
            dist_home = st.number_input("Distance from Home (km)", 1, 30, 9)
            training  = st.slider("Trainings Last Year", 0, 6, 3)

        st.divider()
        st.markdown("**Tenure Details**")
        r4c1, r4c2 = st.columns(2)
        with r4c1:
            yrs_role    = st.number_input("Years in Current Role", 0, 18, 3)
            yrs_promo   = st.number_input("Years Since Last Promotion", 0, 15, 2)
        with r4c2:
            yrs_mgr = st.number_input("Years With Current Manager", 0, 17, 3)

        submitted = st.form_submit_button("🔮 Predict Attrition Risk", type="primary", use_container_width=True)

    if submitted:
        # Build input row
        row = {
            "Age": age, "BusinessTravel": travel, "DailyRate": daily,
            "Department": dept, "DistanceFromHome": dist_home, "Education": edu,
            "EducationField": edu_field, "EnvironmentSatisfaction": env_sat,
            "Gender": gender, "HourlyRate": hourly, "JobInvolvement": job_inv,
            "JobLevel": job_level, "JobRole": role, "JobSatisfaction": job_sat,
            "MaritalStatus": marital, "MonthlyIncome": income,
            "MonthlyRate": monthly_rate, "NumCompaniesWorked": num_companies,
            "OverTime": overtime, "PercentSalaryHike": pct_hike,
            "PerformanceRating": perf, "RelationshipSatisfaction": rel_sat,
            "StockOptionLevel": stock, "TotalWorkingYears": total_years,
            "TrainingTimesLastYear": training, "WorkLifeBalance": wlb,
            "YearsAtCompany": years_company, "YearsInCurrentRole": yrs_role,
            "YearsSinceLastPromotion": yrs_promo, "YearsWithCurrManager": yrs_mgr
        }

        try:
            X_in = pd.DataFrame([{f: row.get(f, 0) for f in feat_names}])[feat_names]
            pred = int(pipeline.predict(X_in)[0])
            prob = pipeline.predict_proba(X_in)[0]
            p_leave = round(prob[1] * 100, 1)
            p_stay  = round(prob[0] * 100, 1)

            # Risk level
            if p_leave >= 70:
                risk, color, emoji = "🔴 HIGH RISK", "#ef4444", "🔴"
            elif p_leave >= 40:
                risk, color, emoji = "🟡 MEDIUM RISK", "#f59e0b", "🟡"
            else:
                risk, color, emoji = "🟢 LOW RISK", "#10b981", "🟢"

            result_label = "Likely to LEAVE" if pred == 1 else "Likely to STAY"

            st.divider()
            st.markdown("## 📊 Prediction Result")

            # Big result card
            st.markdown(f"""
            <div style="background:white;border:3px solid {color};border-radius:16px;padding:28px;text-align:center;margin:16px 0">
              <div style="font-size:48px;margin-bottom:8px">{emoji}</div>
              <div style="font-size:26px;font-weight:700;color:{color}">{result_label}</div>
              <div style="font-size:16px;color:#64748b;margin-top:6px">{risk}</div>
            </div>
            """, unsafe_allow_html=True)

            # Probability bars
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"#### Probability of Leaving: **{p_leave}%**")
                st.progress(p_leave / 100)
            with col2:
                st.markdown(f"#### Probability of Staying: **{p_stay}%**")
                st.progress(p_stay / 100)

            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=p_leave,
                title={"text": "Attrition Risk Score", "font": {"size": 18}},
                number={"suffix": "%"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": color},
                    "steps": [
                        {"range": [0, 40],  "color": "#d1fae5"},
                        {"range": [40, 70], "color": "#fef3c7"},
                        {"range": [70, 100],"color": "#fee2e2"}
                    ],
                    "threshold": {"line": {"color": "red","width": 3}, "value": 70}
                }
            ))
            fig.update_layout(height=280, paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

            # Key risk factors
            st.markdown("### 🔍 Key Risk Factors for This Employee")
            factors = []
            if overtime == "Yes":     factors.append(("⚠️ Working Overtime",         "High", "#ef4444"))
            if p_leave > 50:
                if job_sat <= 2:      factors.append(("😞 Low Job Satisfaction",     "High", "#ef4444"))
                if wlb <= 2:          factors.append(("⚖️  Poor Work-Life Balance",   "High", "#ef4444"))
                if yrs_promo > 4:     factors.append(("📈 No Promotion in 4+ Years", "Medium","#f59e0b"))
                if marital == "Single": factors.append(("💼 Single Employee (higher mobility)", "Low","#0d9488"))
                if num_companies > 3: factors.append(("🏢 High Job-Hopping History", "Medium","#f59e0b"))
                if income < 3000:     factors.append(("💰 Below-Average Income",    "High", "#ef4444"))
            else:
                if job_sat >= 3:      factors.append(("😊 Good Job Satisfaction",   "Positive","#0d9488"))
                if wlb >= 3:          factors.append(("✅ Good Work-Life Balance",   "Positive","#0d9488"))
                if stock > 0:         factors.append(("📦 Has Stock Options",        "Positive","#0d9488"))
                if years_company > 5: factors.append(("🏅 Long Tenure",              "Positive","#0d9488"))

            if factors:
                for label, level, col in factors:
                    st.markdown(f"""
                    <div style="background:white;border-left:4px solid {col};border-radius:8px;padding:10px 16px;margin:6px 0;display:flex;justify-content:space-between;align-items:center;box-shadow:0 1px 4px rgba(0,0,0,.08)">
                      <span style="font-weight:600;color:#0f172a">{label}</span>
                      <span style="background:{col};color:white;padding:2px 10px;border-radius:12px;font-size:12px;font-weight:600">{level}</span>
                    </div>""", unsafe_allow_html=True)

            st.session_state.prediction_made = True
            st.session_state.last_pred = {"label": result_label, "p_leave": p_leave, "risk": risk}

            st.warning("⚠️ **Disclaimer:** This is a statistical model estimate. Use it as one of many HR inputs — not as a definitive conclusion about any individual.")

        except Exception as e:
            st.error(f"Prediction failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
def page_batch():
    st.markdown("# 📁 Batch Prediction")
    st.caption("Upload a CSV with multiple employees to predict attrition for all of them at once.")

    if not os.path.isfile(MODEL_PATH):
        st.warning("Model not trained yet. Go to **Model Training** first.")
        return

    pipeline, meta = load_model()
    feat_names = meta["feature_names"]

    # Template download
    tmpl = pd.DataFrame(columns=feat_names)
    st.download_button("📥 Download CSV Template", tmpl.to_csv(index=False),
                       "employee_template.csv", "text/csv")

    with st.expander("📋 Required columns"):
        st.code(", ".join(feat_names))

    uploaded = st.file_uploader("Upload Employee CSV", type=["csv"])
    if uploaded:
        try:
            df_up = pd.read_csv(uploaded)
            st.markdown(f"**File:** `{uploaded.name}` — {len(df_up):,} rows")

            missing = [f for f in feat_names if f not in df_up.columns]
            if missing:
                st.error(f"❌ Missing columns: `{'`, `'.join(missing)}`")
                return

            if st.button("🚀 Run Batch Prediction", type="primary"):
                X = df_up[feat_names].copy()
                preds = pipeline.predict(X)
                probs = pipeline.predict_proba(X)[:, 1]

                df_up["Predicted_Attrition"] = ["Yes" if p else "No" for p in preds]
                df_up["Leave_Probability_%"]  = (probs * 100).round(1)
                df_up["Risk_Level"] = pd.cut(probs*100, bins=[-1,40,70,101],
                                              labels=["Low","Medium","High"]).astype(str)

                yes = int((preds==1).sum())
                no  = int((preds==0).sum())

                c1,c2,c3 = st.columns(3)
                c1.metric("Total Processed", len(df_up))
                c2.metric("Predicted to Leave", yes)
                c3.metric("Predicted to Stay", no)

                st.dataframe(df_up[["Predicted_Attrition","Leave_Probability_%","Risk_Level"] +
                                    feat_names[:5]], use_container_width=True)

                st.download_button("📥 Download Results CSV",
                                   df_up.to_csv(index=False),
                                   "attrition_predictions.csv", "text/csv",
                                   type="primary")
        except Exception as e:
            st.error(f"Error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
def page_insights():
    st.markdown("# 💡 Analytics & Retention Insights")

    if not os.path.isfile(DATA_PATH):
        st.error("Dataset not found."); return

    df = load_data()
    m  = load_metrics()

    tab1, tab2, tab3 = st.tabs(["🎯 Feature Importance", "📈 Attrition Drivers", "📋 Retention Strategies"])

    with tab1:
        if m and m.get("feature_importances"):
            fi = m["feature_importances"]
            fi_df = pd.DataFrame(list(fi.items()), columns=["Feature","Importance"])
            fi_df["Feature"] = fi_df["Feature"].str.replace("num__","").str.replace("cat__","")
            fi_df = fi_df.sort_values("Importance", ascending=False).head(20)
            fig = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                         color="Importance", color_continuous_scale=["#0891b2","#0d9488","#10b981"],
                         title=f"Top 20 Features — {m['best_model']}")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", yaxis=dict(autorange="reversed"),
                              coloraxis_showscale=False, height=550)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Higher importance = stronger influence on attrition prediction.")
        else:
            st.info("Train the model first to see feature importance.")

    with tab2:
        # Correlation
        df_num = df.copy()
        df_num["Attrition_Num"] = (df["Attrition"]=="Yes").astype(int)
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        corr = df_num[numeric_cols+["Attrition_Num"]].corr()["Attrition_Num"].drop("Attrition_Num").sort_values()
        fig = px.bar(corr.reset_index(), x="Attrition_Num", y="index", orientation="h",
                     color="Attrition_Num", color_continuous_scale="RdYlGn_r",
                     title="Feature Correlation with Attrition (Pearson)")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", yaxis_title="Feature",
                          xaxis_title="Correlation", height=600, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

        # Age vs Attrition
        c1, c2 = st.columns(2)
        with c1:
            fig2 = px.box(df, x="Attrition", y="Age",
                          color="Attrition", color_discrete_map={"Yes":"#ef4444","No":"#0d9488"},
                          title="Age vs Attrition")
            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)
        with c2:
            fig3 = px.box(df, x="Attrition", y="MonthlyIncome",
                          color="Attrition", color_discrete_map={"Yes":"#ef4444","No":"#0d9488"},
                          title="Monthly Income vs Attrition")
            fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", showlegend=False)
            st.plotly_chart(fig3, use_container_width=True)

    with tab3:
        st.markdown("### 🏆 Evidence-Based Retention Strategies")
        actual_rate = round((df["Attrition"]=="Yes").mean()*100, 1)
        ot_rate  = round((df[df["OverTime"]=="Yes"]["Attrition"]=="Yes").mean()*100,1)
        low_sat  = round((df[df["JobSatisfaction"]<=2]["Attrition"]=="Yes").mean()*100,1)
        low_inc  = round((df[df["MonthlyIncome"]<3000]["Attrition"]=="Yes").mean()*100,1)

        strategies = [
            ("💰 Competitive Compensation",
             f"Employees earning < ₹3,000/month have **{low_inc}%** attrition vs overall **{actual_rate}%**. Salaries below market rate are the #1 preventable cause.",
             "Benchmark salaries quarterly. Introduce transparent pay bands."),
            ("⏰ Control Overtime",
             f"Employees with overtime have **{ot_rate}%** attrition. Overwork causes burnout and increases flight risk significantly.",
             "Limit overtime to <10% of workforce. Offer comp-off or premium pay."),
            ("📈 Career Development",
             f"Employees with no promotion in 4+ years have significantly higher attrition. Stagnation kills motivation.",
             "Create clear career ladders. Ensure annual review conversations."),
            ("😊 Job Satisfaction",
             f"Low job satisfaction (1-2) leads to **{low_sat}%** attrition. Disengaged employees are 3x more likely to leave.",
             "Run quarterly pulse surveys. Act on feedback within 30 days."),
            ("🤝 Manager Relationships",
             "Years with current manager is a top negative predictor — good managers retain people.",
             "Train managers on people skills. Monitor manager-level attrition by team."),
            ("📦 Stock Options",
             "Stock option holders show lower attrition. Ownership aligns interests.",
             "Extend stock options to high-performers at all levels."),
        ]

        for title, evidence, action in strategies:
            with st.expander(f"**{title}**", expanded=False):
                st.markdown(f"**📊 Evidence:** {evidence}")
                st.markdown(f"**✅ Action:** {action}")


# ══════════════════════════════════════════════════════════════════════════════
def page_settings():
    st.markdown("# ⚙️ Settings & Account")
    user = st.session_state.get("user", {})

    st.markdown("### 👤 Account Info")
    c1, c2 = st.columns(2)
    c1.markdown(f"**Username:** {user.get('username','N/A')}")
    c1.markdown(f"**Email:** {user.get('email','N/A')}")
    c2.markdown(f"**Joined:** {user.get('created_at','N/A')[:10]}")

    st.divider()
    st.markdown("### 🔧 System Status")
    c1,c2,c3 = st.columns(3)
    c1.metric("Dataset", "✅ Found" if os.path.isfile(DATA_PATH) else "❌ Missing")
    c2.metric("Model",   "✅ Trained" if os.path.isfile(MODEL_PATH) else "⏳ Not trained")
    c3.metric("Metrics", "✅ Available" if os.path.isfile(METRIC_PATH) else "⏳ None")

    if load_metrics():
        m = load_metrics()
        bm = m["models"][m["best_model"]]
        st.divider()
        st.markdown(f"### 🏆 Model: {m['best_model']}")
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Accuracy",  f"{bm['accuracy']:.4f}")
        c2.metric("F1-Score",  f"{bm['f1_score']:.4f}")
        c3.metric("Recall",    f"{bm['recall']:.4f}")
        c4.metric("ROC-AUC",   f"{bm['roc_auc']:.4f}")

    st.divider()
    st.markdown("### 🚪 Logout")
    st.caption("This will clear your session. You'll need to log in again.")
    if st.button("🚪 Sign Out", type="primary"):
        logout()


# ─── Route ─────────────────────────────────────────────────────────────────────
PAGE  = st.session_state.get("page", "🏠  Dashboard")

if   "Dashboard"       in PAGE: page_dashboard()
elif "Dataset"         in PAGE: page_dataset()
elif "Preprocessing"   in PAGE: page_preprocessing()
elif "Training"        in PAGE: page_training()
elif "Predict Attrition" in PAGE: page_predict()
elif "Batch"           in PAGE: page_batch()
elif "Analytics"       in PAGE: page_insights()
elif "Settings"        in PAGE: page_settings()
