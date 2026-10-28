"""
PulseIQ • Healthcare Readmission Analytics & Clinical Intelligence Portal.
Course: Data Engineering and MLOps | Project 5: Hospital Readmission Analytics
Design: 4K Ultra-Fluid 120fps Modern Dark Showcase (JBL & Apple Pro Aesthetic)
"""

import os
import sqlite3
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & 120FPS HARDWARE-ACCELERATED STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="PulseIQ • Clinical Intelligence Portal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS injecting 120fps GPU-accelerated transitions & 4K glassmorphic design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@500;700&display=swap');

    /* 4K Typography Optimization */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #F8FAFC;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
        text-rendering: optimizeLegibility;
    }

    /* Ambient 120fps Aurora Mesh Background */
    .stApp {
        background-color: #06090F;
        background-image: 
            radial-gradient(circle at 10% 10%, rgba(255, 85, 0, 0.12) 0%, transparent 45%),
            radial-gradient(circle at 90% 15%, rgba(6, 182, 212, 0.10) 0%, transparent 40%),
            radial-gradient(circle at 50% 90%, rgba(139, 92, 246, 0.08) 0%, transparent 45%),
            linear-gradient(180deg, rgba(6, 9, 15, 0.95) 0%, rgba(10, 15, 26, 1) 100%);
        background-attachment: fixed;
    }

    /* Smooth Entry Keyframes (120fps Hardware Compositing) */
    @keyframes fadeInUp {
        0% {
            opacity: 0;
            transform: translate3d(0, 24px, 0) scale(0.98);
        }
        100% {
            opacity: 1;
            transform: translate3d(0, 0, 0) scale(1);
        }
    }

    @keyframes pulseGlow {
        0%, 100% {
            box-shadow: 0 0 15px rgba(16, 185, 129, 0.25), inset 0 0 10px rgba(16, 185, 129, 0.15);
        }
        50% {
            box-shadow: 0 0 25px rgba(16, 185, 129, 0.55), inset 0 0 15px rgba(16, 185, 129, 0.35);
        }
    }

    @keyframes shimmerBeam {
        0% { background-position: -200% 0; }
        100% { background-position: 200% 0; }
    }

    /* 4K Hero Brand Banner */
    .hero-container {
        position: relative;
        background: linear-gradient(135deg, rgba(17, 24, 39, 0.85) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 24px;
        padding: 28px 36px;
        margin-bottom: 24px;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.7), inset 0 1px 0 rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(24px);
        -webkit-backdrop-filter: blur(24px);
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 20px;
        animation: fadeInUp 0.6s cubic-bezier(0.16, 1, 0.3, 1) both;
        overflow: hidden;
    }

    .hero-container::after {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, #FF5500, #06B6D4, #8B5CF6, transparent);
        background-size: 200% 100%;
        animation: shimmerBeam 6s linear infinite;
    }

    .hero-title-box h1 {
        font-size: 2.15rem;
        font-weight: 900;
        letter-spacing: -0.035em;
        margin: 0;
        color: #FFFFFF;
        display: flex;
        align-items: center;
        gap: 14px;
    }

    .hero-brand-pill {
        background: linear-gradient(135deg, #FF5500 0%, #FF8A00 100%);
        color: #FFFFFF;
        padding: 5px 14px;
        border-radius: 12px;
        font-size: 1.05rem;
        font-weight: 900;
        box-shadow: 0 6px 20px rgba(255, 85, 0, 0.45);
        letter-spacing: -0.01em;
    }

    .hero-title-box p {
        font-size: 0.94rem;
        color: #94A3B8;
        margin: 8px 0 0 0;
        font-weight: 500;
        letter-spacing: 0.01em;
    }

    .hero-status-box {
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .live-chip {
        background: rgba(16, 185, 129, 0.12);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.35);
        padding: 8px 18px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        display: flex;
        align-items: center;
        gap: 10px;
        animation: pulseGlow 3s infinite ease-in-out;
    }

    .live-dot-beacon {
        width: 9px;
        height: 9px;
        background-color: #10B981;
        border-radius: 50%;
        box-shadow: 0 0 12px #10B981;
    }

    /* Bento Grid 120fps Cards */
    .bento-grid-card {
        position: relative;
        background: rgba(17, 24, 39, 0.72);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 22px;
        padding: 24px 26px;
        box-shadow: 0 14px 35px -8px rgba(0, 0, 0, 0.6);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        transition: transform 0.35s cubic-bezier(0.16, 1, 0.3, 1), 
                    border-color 0.35s cubic-bezier(0.16, 1, 0.3, 1), 
                    box-shadow 0.35s cubic-bezier(0.16, 1, 0.3, 1);
        will-change: transform, box-shadow;
        overflow: hidden;
        animation: fadeInUp 0.7s cubic-bezier(0.16, 1, 0.3, 1) both;
    }

    .bento-grid-card:hover {
        transform: translate3d(0, -6px, 0) scale(1.015);
        border-color: rgba(255, 107, 0, 0.45);
        box-shadow: 0 22px 45px -10px rgba(255, 85, 0, 0.25), 0 0 20px rgba(255, 85, 0, 0.15);
    }

    .card-top {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 10px;
    }

    .card-title {
        font-size: 0.8rem;
        font-weight: 700;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    .card-icon-badge {
        width: 36px;
        height: 36px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.1rem;
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    .card-metric {
        font-size: 2.35rem;
        font-weight: 900;
        color: #FFFFFF;
        letter-spacing: -0.04em;
        line-height: 1.1;
    }

    .card-footer {
        margin-top: 12px;
        font-size: 0.82rem;
        color: #94A3B8;
        display: flex;
        align-items: center;
        gap: 8px;
        font-weight: 500;
    }

    /* Neon Pill Badges */
    .pill-neon-orange {
        background: rgba(255, 85, 0, 0.18);
        color: #FF7700;
        border: 1px solid rgba(255, 85, 0, 0.4);
        padding: 3px 9px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.75rem;
        box-shadow: 0 0 10px rgba(255, 85, 0, 0.2);
    }

    .pill-neon-cyan {
        background: rgba(6, 182, 212, 0.18);
        color: #22D3EE;
        border: 1px solid rgba(6, 182, 212, 0.4);
        padding: 3px 9px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.75rem;
        box-shadow: 0 0 10px rgba(6, 182, 212, 0.2);
    }

    .pill-neon-emerald {
        background: rgba(16, 185, 129, 0.18);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.4);
        padding: 3px 9px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.75rem;
        box-shadow: 0 0 10px rgba(16, 185, 129, 0.2);
    }

    .pill-neon-rose {
        background: rgba(244, 63, 94, 0.18);
        color: #FB7185;
        border: 1px solid rgba(244, 63, 94, 0.4);
        padding: 3px 9px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.75rem;
        box-shadow: 0 0 10px rgba(244, 63, 94, 0.2);
    }

    /* Content Showcase Box */
    .showcase-box {
        background: rgba(17, 24, 39, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 24px;
        padding: 28px 32px;
        margin-bottom: 24px;
        box-shadow: 0 20px 45px -10px rgba(0, 0, 0, 0.6);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        animation: fadeInUp 0.8s cubic-bezier(0.16, 1, 0.3, 1) both;
    }

    .showcase-title {
        font-size: 1.32rem;
        font-weight: 800;
        color: #FFFFFF;
        margin: 0 0 6px 0;
        letter-spacing: -0.02em;
    }

    .showcase-subtitle {
        font-size: 0.9rem;
        color: #94A3B8;
        margin: 0 0 22px 0;
    }

    /* Tabs 120fps Smooth Navigation */
    div[data-baseweb="tab-list"] {
        gap: 10px;
        background-color: rgba(17, 24, 39, 0.95);
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 8px;
        border-radius: 18px;
        margin-bottom: 24px;
        box-shadow: 0 12px 30px -8px rgba(0, 0, 0, 0.5);
    }

    div[data-baseweb="tab"] {
        border-radius: 12px;
        padding: 10px 22px;
        font-weight: 700;
        font-size: 0.93rem;
        color: #94A3B8;
        background: transparent;
        border: none;
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
        will-change: transform, background, color;
    }

    div[data-baseweb="tab"]:hover {
        color: #FFFFFF;
        background: rgba(255, 255, 255, 0.05);
        transform: translateY(-2px);
    }

    div[aria-selected="true"] {
        background: linear-gradient(135deg, #FF5500 0%, #FF8A00 100%) !important;
        color: #FFFFFF !important;
        box-shadow: 0 6px 22px rgba(255, 85, 0, 0.45) !important;
        font-weight: 800 !important;
        transform: scale(1.02);
    }

    /* Sidebar Clean Styling */
    section[data-testid="stSidebar"] {
        background-color: #0A0E17 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }

    span[data-baseweb="tag"] {
        background-color: rgba(255, 85, 0, 0.16) !important;
        border: 1px solid rgba(255, 85, 0, 0.45) !important;
        border-radius: 8px !important;
        color: #FF8A00 !important;
        font-weight: 700 !important;
        transition: all 0.2s ease;
    }
    span[data-baseweb="tag"] span {
        color: #F8FAFC !important;
    }

    /* Complete Removal of Deploy Button, Header Toolbar, and Streamlit Watermark */
    #MainMenu {visibility: hidden !important; display: none !important;}
    header {visibility: hidden !important; display: none !important;}
    header[data-testid="stHeader"] {visibility: hidden !important; display: none !important;}
    [data-testid="stHeader"] {visibility: hidden !important; display: none !important;}
    [data-testid="stToolbar"] {visibility: hidden !important; display: none !important;}
    [data-testid="stToolbarActions"] {visibility: hidden !important; display: none !important;}
    .stDeployButton, [data-testid="stAppDeployButton"] {visibility: hidden !important; display: none !important;}
    button[title="Deploy this app"] {visibility: hidden !important; display: none !important;}

    footer {visibility: hidden !important; display: none !important;}
    [data-testid="stFooter"] {visibility: hidden !important; display: none !important;}
    [data-testid="stStatusWidget"] {visibility: hidden !important; display: none !important;}
    [class*="viewerBadge"] {visibility: hidden !important; display: none !important;}
    [class*="profileContainer"] {visibility: hidden !important; display: none !important;}
    [class*="_profileContainer_"] {visibility: hidden !important; display: none !important;}
    [class*="_container_gzrqn"] {visibility: hidden !important; display: none !important;}
    div[data-testid="manage-app-button"] {visibility: hidden !important; display: none !important;}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. DATA LAYER WITH CACHING
# -----------------------------------------------------------------------------
@st.cache_data(ttl=600)
def load_data():
    """Loads healthcare mart data from SQLite database or fallback CSV."""
    db_path = "data/healthcare.db"
    csv_path = "data/curated/healthcare_mart.csv"

    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        query = """
            SELECT 
                encounter_sk, patient_sk, race, gender, age_bracket,
                admission_type_desc, discharge_disposition_desc, admission_source_desc,
                time_in_hospital, payer_code, medical_specialty,
                num_lab_procedures, num_procedures, num_medications,
                number_outpatient, number_emergency, number_inpatient, prior_admissions,
                diag_1_category, diag_2_category, diag_3_category, number_diagnoses,
                max_glu_serum, a1c_result, metformin, insulin,
                change_in_meds, diabetes_med_prescribed,
                raw_readmitted, readmitted_30d, readmitted_any, readmission_category
            FROM healthcare_mart
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
    elif os.path.exists(csv_path):
        df = pd.read_csv(csv_path, low_memory=False)
    else:
        st.error("Healthcare Mart not found. Please run `python run_pipeline.py`.")
        return pd.DataFrame()

    return df

df = load_data()

# -----------------------------------------------------------------------------
# 3. 4K HERO BANNER
# -----------------------------------------------------------------------------
st.markdown("""
<div class="hero-container">
    <div class="hero-title-group">
        <h1><span class="hero-brand-pill">PULSE IQ</span> Clinical Readmission Intelligence</h1>
        <p>Enterprise Healthcare Data Engineering Pipeline & Analytics Mart • 101,766 Raw Encounters Processed</p>
    </div>
    <div class="hero-status-box">
        <div class="live-chip">
            <span class="live-dot-beacon"></span>
            Live Mart • 99,340 Encounters
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

if df.empty:
    st.info("Please initialize database by executing `python run_pipeline.py`.")
    st.stop()

# -----------------------------------------------------------------------------
# 4. SIDEBAR FILTER ENGINE
# -----------------------------------------------------------------------------
st.sidebar.markdown("### 🎛️ Clinical Filter Controls")
st.sidebar.caption("Dynamic multi-dimensional cohort slicing with real-time reactive updates.")

# Age Filter
all_ages = sorted(df["age_bracket"].dropna().unique().tolist())
selected_ages = st.sidebar.multiselect("Chronological Age Bracket:", all_ages, default=all_ages)

# Diagnosis Filter
all_diags = sorted(df["diag_1_category"].dropna().unique().tolist())
selected_diags = st.sidebar.multiselect("Primary ICD-9 Clinical Chapter:", all_diags, default=all_diags)

# Medication Adjustment Filter
med_change = st.sidebar.selectbox("Medication Dosage Adjusted in Stay:", ["All Encounters", "Yes (Dosage Altered)", "No (Unchanged)"])

# Discharge Disposition Filter
top_discharges = df["discharge_disposition_desc"].value_counts().head(6).index.tolist()
selected_discharges = st.sidebar.multiselect("Discharge Destination:", top_discharges, default=top_discharges[:4])

# Filter Application
filtered_df = df[
    (df["age_bracket"].isin(selected_ages)) &
    (df["diag_1_category"].isin(selected_diags))
]

if med_change == "Yes (Dosage Altered)":
    filtered_df = filtered_df[filtered_df["change_in_meds"] == "Yes"]
elif med_change == "No (Unchanged)":
    filtered_df = filtered_df[filtered_df["change_in_meds"] == "No"]

if selected_discharges:
    filtered_df = filtered_df[filtered_df["discharge_disposition_desc"].isin(selected_discharges)]

# -----------------------------------------------------------------------------
# 5. BENTO GRID 120FPS KPI ROW
# -----------------------------------------------------------------------------
total_enc = len(filtered_df)
uniq_pat = filtered_df["patient_sk"].nunique() if total_enc > 0 else 0
readmit_30d_cnt = filtered_df["readmitted_30d"].sum() if total_enc > 0 else 0
rate_30d = (readmit_30d_cnt / total_enc * 100) if total_enc > 0 else 0
rate_any = (filtered_df["readmitted_any"].sum() / total_enc * 100) if total_enc > 0 else 0
avg_stay = filtered_df["time_in_hospital"].mean() if total_enc > 0 else 0

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.markdown(f"""
    <div class="bento-grid-card">
        <div class="card-top">
            <span class="card-title">Total Encounters</span>
            <span class="card-icon-badge">🏥</span>
        </div>
        <div class="card-metric">{total_enc:,}</div>
        <div class="card-footer"><span class="pill-neon-cyan">Audited</span> Inpatient stays</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="bento-grid-card">
        <div class="card-top">
            <span class="card-title">Unique Patients</span>
            <span class="card-icon-badge">👤</span>
        </div>
        <div class="card-metric">{uniq_pat:,}</div>
        <div class="card-footer"><span class="pill-neon-cyan">HIPAA</span> Surrogate Keys</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="bento-grid-card" style="border-color: rgba(255, 85, 0, 0.3);">
        <div class="card-top">
            <span class="card-title">30-Day Readmission</span>
            <span class="card-icon-badge" style="background: rgba(255, 85, 0, 0.15); color: #FF5500;">🚨</span>
        </div>
        <div class="card-metric" style="color: #FF5500;">{rate_30d:.1f}%</div>
        <div class="card-footer"><span class="pill-neon-orange">{readmit_30d_cnt:,}</span> Acute returns</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="bento-grid-card">
        <div class="card-top">
            <span class="card-title">Overall Readmission</span>
            <span class="card-icon-badge">🔄</span>
        </div>
        <div class="card-metric" style="color: #22D3EE;">{rate_any:.1f}%</div>
        <div class="card-footer"><span class="pill-neon-cyan">All-time</span> Total recurring</div>
    </div>
    """, unsafe_allow_html=True)

with c5:
    st.markdown(f"""
    <div class="bento-grid-card">
        <div class="card-top">
            <span class="card-title">Avg Length of Stay</span>
            <span class="card-icon-badge">⏱️</span>
        </div>
        <div class="card-metric">{avg_stay:.1f} <span style="font-size: 1.1rem; color: #94A3B8; font-weight: 600;">days</span></div>
        <div class="card-footer"><span class="pill-neon-emerald">Mean</span> Hospitalization</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 6. PLOTLY 4K ULTRA-SMOOTH TEMPLATE
# -----------------------------------------------------------------------------
PLOTLY_4K_THEME = dict(
    template="plotly_dark",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Plus Jakarta Sans, sans-serif", size=11.5, color="#E2E8F0"),
    xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", linecolor="rgba(255,255,255,0.12)", zeroline=False),
    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", linecolor="rgba(255,255,255,0.12)", zeroline=False),
    hoverlabel=dict(bgcolor="#0F172A", bordercolor="rgba(255, 85, 0, 0.4)", font=dict(family="Plus Jakarta Sans", size=13, color="#FFFFFF"))
)

# -----------------------------------------------------------------------------
# 7. INTERACTIVE VISUAL INDICATOR TABS
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 1. Age & Demographics",
    "🩺 2. Diagnoses & Specialties",
    "⏱️ 3. Length of Stay Risk",
    "🚪 4. Inflow & Discharge Flows",
    "🚨 5. High-Risk Cohort Explorer",
    "🔮 6. MLOps Predictor"
])

# Clean mapping for consistent, concise outcome labels
OUTCOME_MAP = {
    "<30 Days (High Risk)": "<30d (High Risk)",
    ">30 Days (Late Readmission)": ">30d (Late)",
    "No Readmission": "No Readmission"
}
OUTCOME_COLORS = {
    "<30d (High Risk)": "#FF5500",
    ">30d (Late)": "#F59E0B",
    "No Readmission": "#10B981"
}

# ==================== VIEW 1: AGE & DEMOGRAPHICS ====================
with tab1:
    st.markdown("""
    <div class="showcase-box">
        <div class="showcase-title">1. Readmission Dynamics Across Age Brackets & Demographics</div>
        <div class="showcase-subtitle">Evaluates acute 30-day readmissions vs. overall hospital return frequencies across 10-year age groups.</div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        age_agg = filtered_df.groupby("age_bracket").agg(
            total=("encounter_sk", "count"),
            readmit_30d=("readmitted_30d", "sum"),
            readmit_any=("readmitted_any", "sum")
        ).reset_index()
        age_agg["rate_30d"] = (age_agg["readmit_30d"] / age_agg["total"]) * 100
        age_agg["rate_any"] = (age_agg["readmit_any"] / age_agg["total"]) * 100

        fig_age = go.Figure()
        fig_age.add_trace(go.Bar(
            x=age_agg["age_bracket"],
            y=age_agg["rate_30d"],
            name="30-Day Acute (<30d)",
            marker=dict(color="#FF5500", line=dict(width=0))
        ))
        fig_age.add_trace(go.Bar(
            x=age_agg["age_bracket"],
            y=age_agg["rate_any"],
            name="Overall Return",
            marker=dict(color="#06B6D4", line=dict(width=0))
        ))
        fig_age.update_layout(
            barmode="group",
            title=dict(text="Readmission Rate Across Age Brackets", font=dict(size=13, color="#FFFFFF"), x=0, y=0.98),
            xaxis_title="Age Bracket",
            yaxis_title="Rate (%)",
            legend=dict(title_text="", orientation="h", yanchor="bottom", y=1.04, xanchor="right", x=1.0),
            margin=dict(l=48, r=16, t=65, b=45),
            transition=dict(duration=600, easing="cubic-in-out"),
            **PLOTLY_4K_THEME
        )
        st.plotly_chart(fig_age, use_container_width=True)

    with col2:
        demo_agg = filtered_df.groupby(["race", "readmission_category"]).size().reset_index(name="count")
        demo_agg["Outcome"] = demo_agg["readmission_category"].map(lambda x: OUTCOME_MAP.get(x, x))
        fig_demo = px.bar(
            demo_agg,
            x="race",
            y="count",
            color="Outcome",
            barmode="stack",
            title="Demographic Encounter Distribution",
            color_discrete_map=OUTCOME_COLORS
        )
        fig_demo.update_layout(
            title=dict(text="Demographic Encounter Distribution", font=dict(size=13, color="#FFFFFF"), x=0, y=0.98),
            xaxis_title="Demographic Group",
            yaxis_title="Encounters",
            legend=dict(title_text="", orientation="h", yanchor="bottom", y=1.04, xanchor="right", x=1.0),
            margin=dict(l=55, r=16, t=65, b=45),
            transition=dict(duration=600, easing="cubic-in-out"),
            **PLOTLY_4K_THEME
        )
        st.plotly_chart(fig_demo, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ==================== VIEW 2: DIAGNOSIS & CLINICAL SPECIALTY ====================
with tab2:
    st.markdown("""
    <div class="showcase-box">
        <div class="showcase-title">2. Clinical Diagnostic Chapters & Departmental Readmission Risk</div>
        <div class="showcase-subtitle">Abstracted ICD-9 clinical categories cross-analyzed with physician specialty departments.</div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        diag_agg = filtered_df.groupby("diag_1_category").agg(
            total=("encounter_sk", "count"),
            readmit_30d=("readmitted_30d", "sum")
        ).reset_index()
        diag_agg["rate_30d"] = (diag_agg["readmit_30d"] / diag_agg["total"]) * 100
        diag_agg = diag_agg.sort_values(by="rate_30d", ascending=True)

        fig_diag = px.bar(
            diag_agg,
            x="rate_30d",
            y="diag_1_category",
            orientation="h",
            title="30-Day Readmission Risk by Diagnostic Chapter",
            color="rate_30d",
            color_continuous_scale=["#1E293B", "#FF5500", "#FF1100"],
            labels={"rate_30d": "30-Day Rate (%)", "diag_1_category": "Clinical Category"}
        )
        fig_diag.update_layout(
            title=dict(text="30-Day Readmission Risk by Clinical Chapter", font=dict(size=13, color="#FFFFFF"), x=0, y=0.98),
            xaxis_title="30-Day Rate (%)",
            yaxis_title="",
            coloraxis_showscale=False,
            margin=dict(l=110, r=20, t=55, b=40),
            transition=dict(duration=600, easing="cubic-in-out"),
            **PLOTLY_4K_THEME
        )
        st.plotly_chart(fig_diag, use_container_width=True)

    with col2:
        spec_df = filtered_df[filtered_df["medical_specialty"] != "Unspecified"]
        top_specs = spec_df["medical_specialty"].value_counts().head(8).index
        spec_agg = spec_df[spec_df["medical_specialty"].isin(top_specs)].groupby("medical_specialty").agg(
            total=("encounter_sk", "count"),
            readmit_30d=("readmitted_30d", "sum")
        ).reset_index()
        spec_agg["rate_30d"] = (spec_agg["readmit_30d"] / spec_agg["total"]) * 100
        spec_agg = spec_agg.sort_values(by="rate_30d", ascending=True)

        fig_spec = px.bar(
            spec_agg,
            x="rate_30d",
            y="medical_specialty",
            orientation="h",
            title="Top Medical Specialties by Readmission Intensity",
            color="rate_30d",
            color_continuous_scale=["#1E293B", "#06B6D4", "#3B82F6"],
            labels={"rate_30d": "30-Day Rate (%)", "medical_specialty": "Department"}
        )
        fig_spec.update_layout(
            title=dict(text="Top Medical Specialties by Readmission Intensity", font=dict(size=13, color="#FFFFFF"), x=0, y=0.98),
            xaxis_title="30-Day Rate (%)",
            yaxis_title="",
            coloraxis_showscale=False,
            margin=dict(l=135, r=20, t=55, b=40),
            transition=dict(duration=600, easing="cubic-in-out"),
            **PLOTLY_4K_THEME
        )
        st.plotly_chart(fig_spec, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ==================== VIEW 3: LENGTH OF STAY ANALYSIS ====================
with tab3:
    st.markdown("""
    <div class="showcase-box">
        <div class="showcase-title">3. Inpatient Stay Duration (LOS) vs. Readmission Escalation</div>
        <div class="showcase-subtitle">Non-linear risk curves demonstrating how extended hospitalizations double readmission probability.</div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        fig_box = px.box(
            filtered_df,
            x="readmission_category",
            y="time_in_hospital",
            color="readmission_category",
            color_discrete_map={
                "<30 Days (High Risk)": "#FF5500",
                ">30 Days (Late Readmission)": "#F59E0B",
                "No Readmission": "#10B981"
            }
        )
        fig_box.update_layout(
            showlegend=False,
            title=dict(text="Hospital Stay Duration by Outcome", font=dict(size=13, color="#FFFFFF"), x=0, y=0.98),
            xaxis_title="Clinical Readmission Category",
            yaxis_title="Hospital Stay (Days)",
            margin=dict(l=48, r=20, t=55, b=45),
            transition=dict(duration=600, easing="cubic-in-out"),
            **PLOTLY_4K_THEME
        )
        st.plotly_chart(fig_box, use_container_width=True)

    with col2:
        los_trend = filtered_df.groupby("time_in_hospital").agg(
            total=("encounter_sk", "count"),
            readmit_30d=("readmitted_30d", "sum")
        ).reset_index()
        los_trend["rate_30d"] = (los_trend["readmit_30d"] / los_trend["total"]) * 100

        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(
            x=los_trend["time_in_hospital"],
            y=los_trend["rate_30d"],
            mode="lines+markers",
            line=dict(color="#FF5500", width=3.5, shape="spline"),
            marker=dict(size=8, color="#FF5500", line=dict(width=2, color="#FFFFFF")),
            fill="tozeroy",
            fillcolor="rgba(255, 85, 0, 0.08)",
            name="30-Day Risk"
        ))
        fig_line.update_layout(
            title=dict(text="Readmission Escalation vs. Inpatient Days", font=dict(size=13, color="#FFFFFF"), x=0, y=0.98),
            xaxis_title="Length of Stay (Days)",
            yaxis_title="30-Day Rate (%)",
            legend=dict(title_text="", orientation="h", yanchor="bottom", y=1.04, xanchor="right", x=1.0),
            margin=dict(l=50, r=16, t=65, b=45),
            transition=dict(duration=600, easing="cubic-in-out"),
            **PLOTLY_4K_THEME
        )
        st.plotly_chart(fig_line, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ==================== VIEW 4: ADMISSION & DISCHARGE PATTERNS ====================
with tab4:
    st.markdown("""
    <div class="showcase-box">
        <div class="showcase-title">4. Patient Inflow Channels & Discharge Destination Flow</div>
        <div class="showcase-subtitle">Compares emergency vs. elective admission channels and transitional post-acute care destinations.</div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        src_map = {
            "Emergency Room": "Emergency Room",
            "Physician Referral": "Physician Referral",
            "Transfer from a hospital": "Hospital Transfer",
            "Transfer from another health care facility": "Other Facility Transfer",
            "Clinic Referral": "Clinic Referral"
        }
        top_src = filtered_df["admission_source_desc"].value_counts().head(5).index
        src_sub = filtered_df[filtered_df["admission_source_desc"].isin(top_src)].copy()
        src_sub["Source"] = src_sub["admission_source_desc"].map(lambda x: src_map.get(x, str(x)[:18]))
        src_sub["Outcome"] = src_sub["readmission_category"].map(lambda x: OUTCOME_MAP.get(x, x))
        
        fig_src = px.histogram(
            src_sub,
            x="Source",
            color="Outcome",
            barmode="group",
            color_discrete_map=OUTCOME_COLORS
        )
        fig_src.update_layout(
            title=dict(text="Admission Sources vs. Readmission Outcome", font=dict(size=13, color="#FFFFFF"), x=0, y=0.98),
            xaxis_title="Admission Channel",
            yaxis_title="Encounters",
            xaxis_tickangle=-16,
            legend=dict(title_text="", orientation="h", yanchor="bottom", y=1.04, xanchor="right", x=1.0),
            margin=dict(l=55, r=16, t=72, b=85),
            transition=dict(duration=600, easing="cubic-in-out"),
            **PLOTLY_4K_THEME
        )
        st.plotly_chart(fig_src, use_container_width=True)

    with col2:
        dst_map = {
            "Discharged to home": "Discharged Home",
            "Discharged/transferred to SNF": "Transferred to SNF",
            "Discharged/transferred to home with home health service": "Home Health Care",
            "Discharged/transferred to another short term hospital": "Short Term Hospital",
            "Discharged/transferred to another rehab fac including rehab units of a hospital .": "Rehab Facility"
        }
        top_dst = filtered_df["discharge_disposition_desc"].value_counts().head(5).index
        dst_sub = filtered_df[filtered_df["discharge_disposition_desc"].isin(top_dst)].copy()
        dst_sub["Destination"] = dst_sub["discharge_disposition_desc"].map(lambda x: dst_map.get(x, str(x)[:18]))
        dst_sub["Outcome"] = dst_sub["readmission_category"].map(lambda x: OUTCOME_MAP.get(x, x))
        
        fig_dst = px.histogram(
            dst_sub,
            x="Destination",
            color="Outcome",
            barmode="stack",
            color_discrete_map=OUTCOME_COLORS
        )
        fig_dst.update_layout(
            title=dict(text="Discharge Destination Impact on Readmission", font=dict(size=13, color="#FFFFFF"), x=0, y=0.98),
            xaxis_title="Discharge Destination",
            yaxis_title="Encounters",
            xaxis_tickangle=-16,
            legend=dict(title_text="", orientation="h", yanchor="bottom", y=1.04, xanchor="right", x=1.0),
            margin=dict(l=55, r=16, t=72, b=85),
            transition=dict(duration=600, easing="cubic-in-out"),
            **PLOTLY_4K_THEME
        )
        st.plotly_chart(fig_dst, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ==================== VIEW 5: HIGH-RISK COHORT EXPLORER ====================
with tab5:
    st.markdown("""
    <div class="showcase-box">
        <div class="showcase-title">5. High-Risk Patient Cohort Identification & Query Engine</div>
        <div class="showcase-subtitle">Isolates fragile chronic patients based on prior inpatient utilization, polypharmacy, and glycemic instability.</div>
    """, unsafe_allow_html=True)

    r1, r2, r3 = st.columns(3)
    with r1:
        min_inpatient = st.slider("Min Prior Inpatient Admissions:", 0, 10, 2)
    with r2:
        min_medications = st.slider("Min Active Medications Administered:", 0, 30, 15)
    with r3:
        a1c_choices = filtered_df["a1c_result"].unique().tolist()
        selected_a1c = st.multiselect("HbA1c Laboratory Result:", a1c_choices, default=a1c_choices)

    cohort = filtered_df[
        (filtered_df["number_inpatient"] >= min_inpatient) &
        (filtered_df["num_medications"] >= min_medications) &
        (filtered_df["a1c_result"].isin(selected_a1c))
    ]

    c_cnt = len(cohort)
    c_rate = (cohort["readmitted_30d"].sum() / c_cnt * 100) if c_cnt > 0 else 0
    pop_rate = (filtered_df["readmitted_30d"].sum() / len(filtered_df) * 100) if len(filtered_df) > 0 else 0
    risk_mult = c_rate / max(pop_rate, 0.01)

    st.markdown(f"""
    <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 16px; padding: 20px 24px; margin-bottom: 22px; display: flex; gap: 32px; align-items: center; box-shadow: 0 10px 25px rgba(0,0,0,0.4);">
        <div><span style="color: #94A3B8; font-size: 0.78rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em;">Cohort Volume</span><br><b style="font-size: 1.5rem; color: #FFFFFF; font-weight: 800;">{c_cnt:,}</b></div>
        <div style="border-left: 1px solid rgba(255,255,255,0.1); height: 40px;"></div>
        <div><span style="color: #94A3B8; font-size: 0.78rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em;">Cohort 30-Day Rate</span><br><b style="font-size: 1.5rem; color: #FF5500; font-weight: 800;">{c_rate:.2f}%</b></div>
        <div style="border-left: 1px solid rgba(255,255,255,0.1); height: 40px;"></div>
        <div><span style="color: #94A3B8; font-size: 0.78rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em;">Population Baseline</span><br><b style="font-size: 1.5rem; color: #22D3EE; font-weight: 800;">{pop_rate:.2f}%</b></div>
        <div style="border-left: 1px solid rgba(255,255,255,0.1); height: 40px;"></div>
        <div><span style="color: #94A3B8; font-size: 0.78rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em;">Risk Multiplier</span><br><span class="pill-neon-rose" style="font-size: 1.05rem;">{risk_mult:.2f}x Baseline</span></div>
    </div>
    """, unsafe_allow_html=True)

    display_fields = [
        "encounter_sk", "patient_sk", "age_bracket", "gender", "time_in_hospital",
        "diag_1_category", "number_inpatient", "num_medications",
        "a1c_result", "insulin", "readmission_category"
    ]
    st.dataframe(
        cohort[display_fields].head(100),
        use_container_width=True,
        hide_index=True
    )

    csv_export = cohort.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Export High-Risk Cohort CSV",
        data=csv_export,
        file_name="high_risk_patient_cohort.csv",
        mime="text/csv"
    )

    st.markdown("</div>", unsafe_allow_html=True)

# ==================== VIEW 6: MLOps PREDICTION ENGINE ====================
with tab6:
    st.markdown("""
    <div class="showcase-box">
        <div class="showcase-title">6. MLOps Real-time Prediction Engine</div>
        <div class="showcase-subtitle">Live API integration with the deployed machine learning model for 30-day readmission risk scoring.</div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 🧑‍⚕️ Patient Clinical Profile")
        with st.form("prediction_form"):
            r_age = st.selectbox("Age Bracket", ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)", "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"], index=6)
            r_diag = st.selectbox("Primary Diagnosis", ["Circulatory", "Diabetes", "Respiratory", "Digestive", "Injury", "Musculoskeletal", "Genitourinary", "Neoplasms", "Other"])
            r_time = st.slider("Time in Hospital (Days)", 1, 14, 3)
            r_meds = st.slider("Number of Medications", 1, 80, 15)
            r_prior = st.slider("Prior Inpatient Admissions", 0, 20, 1)
            r_a1c = st.selectbox("A1C Result", ["None", "Norm", ">7", ">8"])
            r_insulin = st.selectbox("Insulin Administered", ["No", "Up", "Down", "Steady"])
            
            submit_button = st.form_submit_button("🧠 Generate Risk Prediction")
            
    with col2:
        st.markdown("### 🎯 AI Risk Assessment")
        if submit_button:
            import requests
            
            payload = {
                "age_bracket": r_age,
                "diag_1_category": r_diag,
                "time_in_hospital": int(r_time),
                "num_medications": int(r_meds),
                "prior_admissions": int(r_prior),
                "number_inpatient": int(r_prior),
                "a1c_result": r_a1c,
                "insulin": r_insulin,
                "change_in_meds": "Yes" if r_insulin in ["Up", "Down"] else "No",
                "diabetes_med_prescribed": "Yes" if r_insulin != "No" else "No"
            }
            
            result = None
            engine_source = None
            
            # 1. Attempt live FastAPI call with short timeout
            try:
                response = requests.post("http://localhost:8000/predict", json=payload, timeout=1.2)
                if response.status_code == 200:
                    api_result = response.json()
                    prob = float(api_result["risk_score"])
                    risk_lvl = api_result.get("risk_level", "High" if prob > 0.25 else "Low")
                    result = {
                        "risk_score": prob,
                        "risk_level": risk_lvl,
                        "explanation": f"Live FastAPI Model Inference (Confidence: {prob*100:.1f}%). Verified through port 8000 microservice."
                    }
                    engine_source = "🌐 Live FastAPI REST Microservice"
            except Exception:
                # FastAPI not running on port 8000; fall back seamlessly to local pipeline
                pass
            
            # 2. Seamless Local Model Fallback (Zero downtime, no connection errors)
            if result is None:
                try:
                    import joblib
                    model_path = "models/best_model.pkl"
                    if os.path.exists(model_path):
                        local_model = joblib.load(model_path)
                        default_features = {
                            "race": "Caucasian",
                            "gender": "Female",
                            "age_bracket": r_age,
                            "admission_type_desc": "1.0",
                            "discharge_disposition_desc": "Discharged to home",
                            "admission_source_desc": "Emergency Room" if r_prior > 0 else "Physician Referral",
                            "time_in_hospital": int(r_time),
                            "medical_specialty": "Unknown",
                            "num_lab_procedures": 40,
                            "num_procedures": 1,
                            "num_medications": int(r_meds),
                            "number_outpatient": 0,
                            "number_emergency": 0,
                            "number_inpatient": int(r_prior),
                            "prior_admissions": int(r_prior),
                            "diag_1_code": "410",
                            "diag_1_category": r_diag,
                            "diag_2_category": "Diabetes",
                            "diag_3_category": "Other",
                            "number_diagnoses": 5,
                            "max_glu_serum": "None",
                            "a1c_result": r_a1c,
                            "metformin": "No",
                            "insulin": r_insulin,
                            "change_in_meds": "Yes" if r_insulin in ["Up", "Down"] else "No",
                            "diabetes_med_prescribed": "Yes" if r_insulin != "No" else "No"
                        }
                        df_in = pd.DataFrame([default_features])
                        prob = float(local_model.predict_proba(df_in)[0][1])
                        
                        # Clinically sound risk tier
                        if prob >= 0.35:
                            risk_lvl = "High"
                        elif prob >= 0.18:
                            risk_lvl = "Moderate"
                        else:
                            risk_lvl = "Low"
                        
                        # Identify key risk factors for clinical explanation
                        drivers = []
                        if r_prior >= 2:
                            drivers.append(f"{r_prior} prior inpatient stays")
                        if r_time >= 7:
                            drivers.append(f"prolonged hospital stay ({r_time} days)")
                        if r_meds >= 20:
                            drivers.append(f"polypharmacy ({r_meds} medications)")
                        if r_diag in ["Circulatory", "Diabetes", "Respiratory"]:
                            drivers.append(f"high-risk {r_diag} diagnosis")
                        if r_insulin in ["Up", "Down"]:
                            drivers.append("acute insulin dosage titration")
                            
                        driver_text = f" Driven primarily by {', '.join(drivers)}." if drivers else " Patient exhibits manageable clinical baseline."
                        
                        result = {
                            "risk_score": prob,
                            "risk_level": risk_lvl,
                            "explanation": f"ML Model Inference (Confidence: {prob*100:.1f}%).{driver_text}"
                        }
                        engine_source = "⚡ Embedded MLOps Engine (Local Inference Active)"
                except Exception as local_err:
                    st.error(f"Prediction engine error: {str(local_err)}")
            
            if result:
                risk_color = "#FF5500" if result["risk_level"] in ["High", "Critical"] else ("#F59E0B" if result["risk_level"] == "Moderate" else "#10B981")
                
                st.markdown(f"""
                <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid {risk_color}; border-radius: 16px; padding: 26px; margin-top: 10px; text-align: center; box-shadow: 0 10px 30px {risk_color}33;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                        <span style="color: #94A3B8; font-size: 0.8rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em;">Predicted 30-Day Readmission Risk</span>
                        <span style="background: rgba(255,255,255,0.08); color: #CBD5E1; padding: 3px 10px; border-radius: 9999px; font-size: 0.72rem; font-weight: 600;">{engine_source}</span>
                    </div>
                    <div style="font-size: 3.6rem; color: {risk_color}; font-weight: 900; line-height: 1;">{result["risk_score"]*100:.1f}%</div>
                    <div style="margin-top: 14px; display: inline-block;">
                        <span style="background: {risk_color}22; color: {risk_color}; border: 1px solid {risk_color}55; padding: 6px 16px; border-radius: 9999px; font-weight: 800; font-size: 0.9rem; letter-spacing: 0.05em;">
                            {result['risk_level'].upper()} RISK LEVEL
                        </span>
                    </div>
                    <div style="margin-top: 18px; color: #E2E8F0; font-size: 0.92rem; background: rgba(0,0,0,0.25); padding: 12px 16px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.06); text-align: left;">
                        💡 <b>Explainability Insight:</b> {result['explanation']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Gauge chart
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = result["risk_score"] * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Readmission Probability Gauge", 'font': {'color': '#94A3B8', 'size': 13}},
                    gauge = {
                        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "rgba(255,255,255,0.2)"},
                        'bar': {'color': risk_color},
                        'bgcolor': "rgba(0,0,0,0)",
                        'borderwidth': 2,
                        'bordercolor': "rgba(255,255,255,0.1)",
                        'steps': [
                            {'range': [0, 20], 'color': "rgba(16, 185, 129, 0.15)"},
                            {'range': [20, 50], 'color': "rgba(245, 158, 11, 0.15)"},
                            {'range': [50, 100], 'color': "rgba(255, 85, 0, 0.18)"}],
                    }
                ))
                fig.update_layout(height=230, margin=dict(l=10, r=10, t=35, b=10), paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#E2E8F0"))
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("👈 Enter patient details and click Generate to see the MLOps model prediction.")

    st.markdown("</div>", unsafe_allow_html=True)

