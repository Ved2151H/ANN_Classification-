import streamlit as st
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle
import tensorflow as tf
import os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Predictor",
    page_icon="🧠",
    layout="centered",
)

# ── Dark theme (cyan/teal accent, no purple) ──────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=Share+Tech+Mono&display=swap');

html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background-color: #060d0d !important;
    color: #d0f0ec !important;
    font-family: 'Rajdhani', sans-serif !important;
}

[data-testid="stSidebar"] { display: none; }
[data-testid="stHeader"]  { background: transparent !important; }

[data-testid="stMainBlockContainer"] {
    background: #060d0d !important;
    padding-top: 2rem;
}

[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(29,158,117,0.05) 1px, transparent 1px),
        linear-gradient(90deg, rgba(29,158,117,0.05) 1px, transparent 1px);
    background-size: 32px 32px;
    pointer-events: none;
    z-index: 0;
}

h1, h2, h3, h4, label, p {
    font-family: 'Rajdhani', sans-serif !important;
    color: #d0f0ec !important;
}

[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stSelectbox"] select,
div[data-baseweb="select"] {
    background-color: #0b1a18 !important;
    border: 1px solid rgba(29,158,117,0.3) !important;
    border-radius: 8px !important;
    color: #a0e0d0 !important;
    font-family: 'Share Tech Mono', monospace !important;
}

div[data-baseweb="select"] * {
    background-color: #0b1a18 !important;
    color: #a0e0d0 !important;
    font-family: 'Share Tech Mono', monospace !important;
}

[data-testid="stSlider"] > div > div > div > div {
    background: #1D9E75 !important;
}
[data-testid="stSlider"] [data-testid="stTickBarMin"],
[data-testid="stSlider"] [data-testid="stTickBarMax"] {
    color: #3a7060 !important;
    font-family: 'Share Tech Mono', monospace !important;
}

[data-testid="stButton"] > button {
    background: rgba(29,158,117,0.15) !important;
    border: 1px solid #1D9E75 !important;
    border-radius: 8px !important;
    color: #5DCAA5 !important;
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
    background: rgba(29,158,117,0.35) !important;
    border-color: #5DCAA5 !important;
    color: #d0f0ec !important;
}

[data-testid="stAlert"] {
    border-radius: 10px !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 16px !important;
    font-weight: 500 !important;
}

[data-testid="stMetric"] {
    background: rgba(29,158,117,0.08) !important;
    border: 1px solid rgba(29,158,117,0.2) !important;
    border-radius: 10px !important;
    padding: 12px 16px !important;
}
[data-testid="stMetricLabel"] {
    color: #3a7060 !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 1px !important;
}
[data-testid="stMetricValue"] {
    color: #5DCAA5 !important;
    font-family: 'Share Tech Mono', monospace !important;
}

hr {
    border: none !important;
    border-top: 1px solid rgba(29,158,117,0.15) !important;
    margin: 1.2rem 0 !important;
}

[data-testid="stExpander"] {
    background: rgba(29,158,117,0.05) !important;
    border: 1px solid rgba(29,158,117,0.2) !important;
    border-radius: 10px !important;
}
</style>
""", unsafe_allow_html=True)


# ── Header banner ─────────────────────────────────────────────────────────────
st.markdown(
    '<div style="'
    'background:linear-gradient(135deg,rgba(29,158,117,0.15) 0%,rgba(8,80,65,0.10) 100%);'
    'border:1px solid rgba(29,158,117,0.35);'
    'border-top:2px solid #1D9E75;'
    'border-radius:12px;'
    'padding:20px 24px 16px;'
    'margin-bottom:28px;">'
    '<div style="display:flex;align-items:center;gap:12px;">'
    '<span style="font-size:28px;">🧠</span>'
    '<div>'
    '<div style="font-family:Rajdhani,sans-serif;font-size:22px;font-weight:700;'
    'letter-spacing:2px;text-transform:uppercase;color:#d0f0ec;">'
    'Customer Churn Predictor</div>'
    '<div style="font-family:Share Tech Mono,monospace;font-size:11px;color:#3a7060;letter-spacing:1px;">'
    'neural classification engine &middot; v2.1 &middot; model loaded &#10003;</div>'
    '</div>'
    '<div style="margin-left:auto;width:8px;height:8px;background:#1D9E75;border-radius:50%;'
    'box-shadow:0 0 8px #1D9E75;"></div>'
    '</div></div>',
    unsafe_allow_html=True,
)


# ── Load model & encoders ─────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_assets():
    mdl = tf.keras.models.load_model(os.path.join(BASE_DIR, 'model.h5'))
    with open(os.path.join(BASE_DIR, 'label_encoder_gender.pkl'), 'rb') as f:
        le_gender = pickle.load(f)
    with open(os.path.join(BASE_DIR, 'onehot_encoder_geo.pkl'), 'rb') as f:
        ohe_geo = pickle.load(f)
    with open(os.path.join(BASE_DIR, 'scaler.pkl'), 'rb') as f:
        scl = pickle.load(f)
    return mdl, le_gender, ohe_geo, scl

model, label_encoder_gender, onehot_encoder_geo, scaler = load_assets()


# ── Section label helper ──────────────────────────────────────────────────────
def section(title):
    st.markdown(
        f'<div style="font-family:Share Tech Mono,monospace;font-size:10px;'
        f'letter-spacing:2px;text-transform:uppercase;color:#1D9E75;margin:18px 0 6px;">'
        f'&mdash; {title}</div>',
        unsafe_allow_html=True,
    )


# ── Form ──────────────────────────────────────────────────────────────────────
section("Location & Identity")
col1, col2 = st.columns(2)
with col1:
    geography = st.selectbox("Geography", onehot_encoder_geo.categories_[0])
with col2:
    gender = st.selectbox("Gender", label_encoder_gender.classes_)

st.markdown("---")
section("Financial Profile")
col3, col4 = st.columns(2)
with col3:
    credit_score     = st.number_input("Credit Score",         value=600.0,   min_value=300.0, max_value=850.0)
    estimated_salary = st.number_input("Estimated Salary ($)", value=50000.0, min_value=0.0)
with col4:
    balance          = st.number_input("Balance ($)",          value=60000.0, min_value=0.0)
    num_of_products  = st.slider("Number of Products", 1, 4, 2)

st.markdown("---")
section("Customer Behaviour")
col5, col6 = st.columns(2)
with col5:
    age    = st.slider("Age", 18, 92, 40)
with col6:
    tenure = st.slider("Tenure (years)", 0, 10, 3)

st.markdown("---")
section("Account Flags")
col7, col8 = st.columns(2)
with col7:
    has_cr_card      = st.selectbox("Has Credit Card",  [1, 0], format_func=lambda x: "Yes" if x else "No")
with col8:
    is_active_member = st.selectbox("Is Active Member", [1, 0], format_func=lambda x: "Yes" if x else "No")

st.markdown("---")

# ── Metrics preview ───────────────────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
m1.metric("Age",      age)
m2.metric("Balance",  f"${balance:,.0f}")
m3.metric("Products", num_of_products)
m4.metric("Tenure",   f"{tenure} yr")

st.markdown("<br>", unsafe_allow_html=True)

# ── Predict ───────────────────────────────────────────────────────────────────
if st.button("⚡  Run Prediction"):

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
    geo_df_enc  = pd.DataFrame(
        geo_encoded,
        columns=onehot_encoder_geo.get_feature_names_out(['Geography']),
    )

    input_data   = pd.concat([input_data.reset_index(drop=True), geo_df_enc], axis=1)
    input_scaled = scaler.transform(input_data)

    prediction_proba = float(model.predict(input_scaled)[0][0])
    pct              = round(prediction_proba * 100, 1)

    if prediction_proba > 0.5:
        color   = "#e24b4a"
        bg      = "rgba(226,75,74,0.08)"
        border  = "rgba(226,75,74,0.35)"
        divider = "rgba(226,75,74,0.15)"
        icon    = "&#9888;"
        verdict = "LIKELY TO CHURN"
        note    = "Consider a targeted retention offer — loyalty discount, personal outreach, or product review."
    else:
        color   = "#1D9E75"
        bg      = "rgba(29,158,117,0.08)"
        border  = "rgba(29,158,117,0.35)"
        divider = "rgba(29,158,117,0.15)"
        icon    = "&#10003;"
        verdict = "LIKELY TO STAY"
        note    = "Customer profile is stable. Continue standard engagement and monitor periodically."

    bar_html = (
        '<div style="background:rgba(255,255,255,0.07);border-radius:4px;height:6px;overflow:hidden;margin-bottom:6px;">'
        f'<div style="width:{pct}%;height:100%;background:{color};border-radius:4px;"></div>'
        '</div>'
    )

    labels_html = (
        '<div style="display:flex;justify-content:space-between;'
        'font-family:Share Tech Mono,monospace;font-size:10px;color:#3a7060;">'
        '<span>0% &middot; certain to stay</span>'
        '<span>100% &middot; certain to leave</span>'
        '</div>'
    )

    note_html = (
        f'<div style="margin-top:14px;padding-top:12px;border-top:1px solid {divider};'
        'font-family:Share Tech Mono,monospace;font-size:12px;color:#3a8070;line-height:1.6;">'
        f'&gt; {note}'
        '</div>'
    )

    header_html = (
        '<div style="display:flex;align-items:center;gap:14px;margin-bottom:14px;">'
        f'<span style="font-size:28px;color:{color};">{icon}</span>'
        '<div>'
        '<div style="font-family:Share Tech Mono,monospace;font-size:10px;'
        'letter-spacing:1.5px;color:#3a7060;margin-bottom:4px;">PREDICTION OUTPUT</div>'
        f'<div style="font-family:Rajdhani,sans-serif;font-size:20px;font-weight:700;'
        f'color:{color};letter-spacing:1px;">{verdict}</div>'
        '</div>'
        '<div style="margin-left:auto;text-align:right;">'
        '<div style="font-family:Share Tech Mono,monospace;font-size:10px;'
        'letter-spacing:1.5px;color:#3a7060;margin-bottom:2px;">CHURN PROBABILITY</div>'
        f'<div style="font-family:Share Tech Mono,monospace;font-size:32px;font-weight:700;color:{color};">'
        f'{pct}%</div>'
        '</div>'
        '</div>'
    )

    card_html = (
        f'<div style="background:{bg};border:1px solid {border};border-radius:12px;padding:20px 22px;margin-top:8px;">'
        + header_html
        + bar_html
        + labels_html
        + note_html
        + '</div>'
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(card_html, unsafe_allow_html=True)