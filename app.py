import streamlit as st
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle
import tensorflow as tf
import os

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Predictor",
    page_icon="🧠",
    layout="centered",
)

# ── Dark theme injection ──────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=Share+Tech+Mono&display=swap');

/* ── Root / body ── */
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background-color: #0a0a0f !important;
    color: #e0e0f0 !important;
    font-family: 'Rajdhani', sans-serif !important;
}

[data-testid="stSidebar"] { display: none; }
[data-testid="stHeader"]  { background: transparent !important; }

/* ── Main container ── */
[data-testid="stMainBlockContainer"] {
    background: #0a0a0f !important;
    padding-top: 2rem;
}

/* ── Grid background overlay ── */
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(127,119,221,0.04) 1px, transparent 1px),
        linear-gradient(90deg, rgba(127,119,221,0.04) 1px, transparent 1px);
    background-size: 32px 32px;
    pointer-events: none;
    z-index: 0;
}

/* ── Typography ── */
h1, h2, h3, h4, label, p {
    font-family: 'Rajdhani', sans-serif !important;
    color: #e0e0f0 !important;
}

/* ── Inputs / selects ── */
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stSelectbox"] select,
div[data-baseweb="select"] {
    background-color: #111120 !important;
    border: 1px solid rgba(127,119,221,0.25) !important;
    border-radius: 8px !important;
    color: #c8c8f0 !important;
    font-family: 'Share Tech Mono', monospace !important;
}

div[data-baseweb="select"] * {
    background-color: #111120 !important;
    color: #c8c8f0 !important;
    font-family: 'Share Tech Mono', monospace !important;
}

/* ── Slider ── */
[data-testid="stSlider"] > div > div > div > div {
    background: #534AB7 !important;
}
[data-testid="stSlider"] [data-testid="stTickBarMin"],
[data-testid="stSlider"] [data-testid="stTickBarMax"] {
    color: #7070a0 !important;
    font-family: 'Share Tech Mono', monospace !important;
}

/* ── Buttons ── */
[data-testid="stButton"] > button {
    background: rgba(83,74,183,0.2) !important;
    border: 1px solid #534AB7 !important;
    border-radius: 8px !important;
    color: #afa9ec !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    width: 100% !important;
    padding: 0.6rem !important;
    transition: all 0.2s !important;
}
[data-testid="stButton"] > button:hover {
    background: rgba(83,74,183,0.45) !important;
    border-color: #7f77dd !important;
    color: #e8e8ff !important;
}

/* ── Success / warning boxes ── */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 16px !important;
    font-weight: 500 !important;
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: rgba(127,119,221,0.08) !important;
    border: 1px solid rgba(127,119,221,0.2) !important;
    border-radius: 10px !important;
    padding: 12px 16px !important;
}
[data-testid="stMetricLabel"] {
    color: #7070a0 !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 1px !important;
}
[data-testid="stMetricValue"] {
    color: #afa9ec !important;
    font-family: 'Share Tech Mono', monospace !important;
}

/* ── Section divider ── */
hr {
    border: none !important;
    border-top: 1px solid rgba(127,119,221,0.15) !important;
    margin: 1.2rem 0 !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: rgba(127,119,221,0.05) !important;
    border: 1px solid rgba(127,119,221,0.2) !important;
    border-radius: 10px !important;
}
</style>
""", unsafe_allow_html=True)


# ── Header banner ─────────────────────────────────────────────────────────────
st.markdown("""
<div style="
    background: linear-gradient(135deg, rgba(83,74,183,0.18) 0%, rgba(29,158,117,0.10) 100%);
    border: 1px solid rgba(127,119,221,0.3);
    border-top: 2px solid #7f77dd;
    border-radius: 12px;
    padding: 20px 24px 16px;
    margin-bottom: 28px;
    position: relative;
">
    <div style="display:flex; align-items:center; gap:12px;">
        <span style="font-size:28px;">🧠</span>
        <div>
            <div style="
                font-family:'Rajdhani',sans-serif;
                font-size:22px; font-weight:700;
                letter-spacing:2px; text-transform:uppercase;
                color:#e8e8ff;
            ">Customer Churn Predictor</div>
            <div style="
                font-family:'Share Tech Mono',monospace;
                font-size:11px; color:#7070a0; letter-spacing:1px;
            ">neural classification engine · v2.1 · model loaded ✓</div>
        </div>
        <div style="
            margin-left:auto; width:8px; height:8px;
            background:#1D9E75; border-radius:50%;
            box-shadow:0 0 8px #1D9E75;
        "></div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Load model & encoders ─────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model(os.path.join(BASE_DIR, 'model.h5'))
    with open(os.path.join(BASE_DIR, 'label_encoder_gender.pkl'), 'rb') as f:
        le_gender = pickle.load(f)
    with open(os.path.join(BASE_DIR, 'onehot_encoder_geo.pkl'), 'rb') as f:
        ohe_geo = pickle.load(f)
    with open(os.path.join(BASE_DIR, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    return model, le_gender, ohe_geo, scaler

model, label_encoder_gender, onehot_encoder_geo, scaler = load_assets()


# ── Section label helper ───────────────────────────────────────────────────────
def section(title):
    st.markdown(f"""
    <div style="
        font-family:'Share Tech Mono',monospace;
        font-size:10px; letter-spacing:2px;
        text-transform:uppercase; color:#534AB7;
        margin: 18px 0 6px;
    ">— {title}</div>
    """, unsafe_allow_html=True)


# ── Form ──────────────────────────────────────────────────────────────────────
section("Location & Identity")
col1, col2 = st.columns(2)
with col1:
    geography = st.selectbox("Geography", onehot_encoder_geo.categories_[0], label_visibility="visible")
with col2:
    gender = st.selectbox("Gender", label_encoder_gender.classes_, label_visibility="visible")

st.markdown("---")
section("Financial Profile")
col3, col4 = st.columns(2)
with col3:
    credit_score    = st.number_input("Credit Score",    value=600.0,   min_value=300.0, max_value=850.0)
    estimated_salary= st.number_input("Estimated Salary ($)", value=50000.0, min_value=0.0)
with col4:
    balance         = st.number_input("Balance ($)",     value=60000.0, min_value=0.0)
    num_of_products = st.slider("Number of Products", 1, 4, 2)

st.markdown("---")
section("Customer Behaviour")
col5, col6 = st.columns(2)
with col5:
    age    = st.slider("Age",    18, 92, 40)
with col6:
    tenure = st.slider("Tenure (years)", 0, 10, 3)

st.markdown("---")
section("Account Flags")
col7, col8 = st.columns(2)
with col7:
    has_cr_card     = st.selectbox("Has Credit Card",  [1, 0], format_func=lambda x: "Yes" if x else "No")
with col8:
    is_active_member= st.selectbox("Is Active Member", [1, 0], format_func=lambda x: "Yes" if x else "No")

st.markdown("---")

# ── Metrics preview ───────────────────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
m1.metric("Age",      age)
m2.metric("Balance",  f"${balance:,.0f}")
m3.metric("Products", num_of_products)
m4.metric("Tenure",   f"{tenure} yr")

st.markdown("<br>", unsafe_allow_html=True)

# ── Predict button ────────────────────────────────────────────────────────────
if st.button("⚡  Run Prediction"):

    # Build input DataFrame
    input_data = pd.DataFrame({
        'CreditScore':     [credit_score],
        'Gender':          [label_encoder_gender.transform([gender])[0]],
        'Age':             [age],
        'Tenure':          [tenure],
        'Balance':         [balance],
        'NumOfProducts':   [num_of_products],
        'HasCrCard':       [has_cr_card],
        'IsActiveMember':  [is_active_member],
        'EstimatedSalary': [estimated_salary],
    })

    geo_df      = pd.DataFrame([[geography]], columns=['Geography'])
    geo_encoded = onehot_encoder_geo.transform(geo_df).toarray()
    geo_df_enc  = pd.DataFrame(geo_encoded,
                               columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

    input_data  = pd.concat([input_data.reset_index(drop=True), geo_df_enc], axis=1)
    input_scaled= scaler.transform(input_data)

    prediction      = model.predict(input_scaled)
    prediction_proba= float(prediction[0][0])
    pct             = round(prediction_proba * 100, 1)

    # ── Result display ──
    st.markdown("<br>", unsafe_allow_html=True)

    if prediction_proba > 0.5:
        color  = "#e24b4a"
        bg     = "rgba(226,75,74,0.08)"
        border = "rgba(226,75,74,0.35)"
        icon   = "⚠️"
        verdict= "LIKELY TO CHURN"
        note   = "Consider a targeted retention offer — loyalty discount, personal outreach, or product review."
    else:
        color  = "#1D9E75"
        bg     = "rgba(29,158,117,0.08)"
        border = "rgba(29,158,117,0.35)"
        icon   = "✅"
        verdict= "LIKELY TO STAY"
        note   = "Customer profile is stable. Continue standard engagement and monitor periodically."

    # Result card
    st.markdown(f"""
    <div style="
        background:{bg};
        border:1px solid {border};
        border-radius:12px;
        padding:20px 22px;
        margin-top:8px;
    ">
        <div style="display:flex; align-items:center; gap:14px; margin-bottom:14px;">
            <span style="font-size:28px;">{icon}</span>
            <div>
                <div style="font-family:'Share Tech Mono',monospace; font-size:10px;
                            letter-spacing:1.5px; color:#7070a0; margin-bottom:4px;">
                    PREDICTION OUTPUT
                </div>
                <div style="font-family:'Rajdhani',sans-serif; font-size:20px;
                            font-weight:700; color:{color}; letter-spacing:1px;">
                    {verdict}
                </div>
            </div>
            <div style="margin-left:auto; text-align:right;">
                <div style="font-family:'Share Tech Mono',monospace; font-size:10px;
                            letter-spacing:1.5px; color:#7070a0; margin-bottom:2px;">
                    CHURN PROBABILITY
                </div>
                <div style="font-family:'Share Tech Mono',monospace;
                            font-size:32px; font-weight:700; color:{color};">
                    {pct}%
                </div>
            </div>
        </div>

        <!-- probability bar -->
        <div style="background:rgba(255,255,255,0.07); border-radius:4px; height:6px; overflow:hidden; margin-bottom:6px;">
            <div style="width:{pct}%; height:100%; background:{color}; border-radius:4px;
                        transition:width 0.8s ease;"></div>
        </div>
        <div style="display:flex; justify-content:space-between;
                    font-family:'Share Tech Mono',monospace; font-size:10px; color:#5050a0;">
            <span>0% · certain to stay</span><span>100% · certain to leave</span>
        </div>

        <div style="margin-top:14px; padding-top:12px;
                    border-top:1px solid rgba(127,119,221,0.15);
                    font-family:'Share Tech Mono',monospace; font-size:12px;
                    color:#9090c0; line-height:1.6;">
            &gt; {note}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Streamlit native alert as backup fallback
    if prediction_proba > 0.5:
        st.warning(f"Churn probability: **{pct}%** — customer is at risk.")
    else:
        st.success(f"Churn probability: **{pct}%** — customer is stable.")