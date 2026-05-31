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
    layout="wide",
)

# ── Pure black theme ──────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=Share+Tech+Mono&display=swap');

/* ── Full black background everywhere ── */
html, body,
[data-testid="stApp"],
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"],
[data-testid="block-container"],
section.main, .main .block-container {
    background-color: #000000 !important;
    color: #e8e8e8 !important;
    font-family: 'Rajdhani', sans-serif !important;
}

/* ── Remove Streamlit chrome ── */
[data-testid="stSidebar"]          { display: none !important; }
[data-testid="stHeader"]           { background: transparent !important; }
[data-testid="stToolbar"]          { display: none !important; }
[data-testid="stDecoration"]       { display: none !important; }
#MainMenu, footer, header          { visibility: hidden !important; }

/* ── Full width, no max-width cap ── */
[data-testid="block-container"],
.block-container {
    max-width: 100% !important;
    padding: 1.5rem 3rem !important;
}

/* ── Subtle grid overlay ── */
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(255,255,255,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.03) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
}

/* ── Typography ── */
h1, h2, h3, h4, label, p, div {
    font-family: 'Rajdhani', sans-serif !important;
    color: #e8e8e8 !important;
}

/* ── Number inputs ── */
[data-testid="stNumberInput"] input {
    background-color: #111111 !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 8px !important;
    color: #e8e8e8 !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 15px !important;
}
[data-testid="stNumberInput"] input:focus {
    border-color: #555555 !important;
    box-shadow: none !important;
}

/* ── Selectbox ── */
div[data-baseweb="select"] > div {
    background-color: #111111 !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 8px !important;
}
div[data-baseweb="select"] * {
    background-color: #111111 !important;
    color: #e8e8e8 !important;
    font-family: 'Share Tech Mono', monospace !important;
}
[data-baseweb="popover"] [role="option"]:hover {
    background-color: #1e1e1e !important;
}

/* ── Slider ── */
[data-testid="stSlider"] > div > div > div > div {
    background: #ffffff !important;
}
[data-testid="stSlider"] [data-testid="stTickBarMin"],
[data-testid="stSlider"] [data-testid="stTickBarMax"] {
    color: #555555 !important;
    font-family: 'Share Tech Mono', monospace !important;
}
[data-testid="stSlider"] p {
    color: #aaaaaa !important;
}

/* ── Button ── */
[data-testid="stButton"] > button {
    background: #111111 !important;
    border: 1px solid #333333 !important;
    border-radius: 8px !important;
    color: #e8e8e8 !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    letter-spacing: 3px !important;
    text-transform: uppercase !important;
    width: 100% !important;
    padding: 0.75rem !important;
    transition: all 0.2s !important;
}
[data-testid="stButton"] > button:hover {
    background: #1a1a1a !important;
    border-color: #666666 !important;
    color: #ffffff !important;
}
[data-testid="stButton"] > button:active {
    transform: scale(0.99) !important;
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: #0d0d0d !important;
    border: 1px solid #222222 !important;
    border-radius: 10px !important;
    padding: 14px 18px !important;
}
[data-testid="stMetricLabel"] > div {
    color: #555555 !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 1px !important;
}
[data-testid="stMetricValue"] > div {
    color: #e8e8e8 !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 22px !important;
}

/* ── Divider ── */
hr {
    border: none !important;
    border-top: 1px solid #1a1a1a !important;
    margin: 1rem 0 !important;
}

/* ── Responsive columns: stack on narrow screens ── */
@media (max-width: 768px) {
    [data-testid="block-container"], .block-container {
        padding: 1rem 1rem !important;
    }
    [data-testid="column"] {
        min-width: 100% !important;
        flex: 1 1 100% !important;
    }
}
</style>
""", unsafe_allow_html=True)


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
        f'letter-spacing:2px;text-transform:uppercase;color:#555555;margin:22px 0 8px;">'
        f'&mdash; {title}</div>',
        unsafe_allow_html=True,
    )


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    '<div style="'
    'background:#0a0a0a;'
    'border:1px solid #1e1e1e;'
    'border-left:3px solid #ffffff;'
    'border-radius:10px;'
    'padding:22px 28px 18px;'
    'margin-bottom:32px;">'
    '<div style="display:flex;align-items:center;gap:14px;flex-wrap:wrap;">'
    '<span style="font-size:30px;">🧠</span>'
    '<div style="flex:1;min-width:200px;">'
    '<div style="font-family:Rajdhani,sans-serif;font-size:24px;font-weight:700;'
    'letter-spacing:3px;text-transform:uppercase;color:#ffffff;">'
    'Customer Churn Predictor</div>'
    '<div style="font-family:Share Tech Mono,monospace;font-size:11px;color:#444444;letter-spacing:1px;">'
    'neural classification engine &middot; v2.1 &middot; model loaded &#10003;</div>'
    '</div>'
    '<div style="display:flex;align-items:center;gap:8px;">'
    '<div style="width:7px;height:7px;background:#ffffff;border-radius:50%;opacity:0.8;"></div>'
    '<span style="font-family:Share Tech Mono,monospace;font-size:11px;color:#444444;">ONLINE</span>'
    '</div>'
    '</div></div>',
    unsafe_allow_html=True,
)


# ── Form: 3-column layout on wide screens ─────────────────────────────────────
section("Location & Identity")
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    geography = st.selectbox("Geography", onehot_encoder_geo.categories_[0])
with col2:
    gender = st.selectbox("Gender", label_encoder_gender.classes_)
with col3:
    has_cr_card = st.selectbox("Has Credit Card", [1, 0], format_func=lambda x: "Yes" if x else "No")

st.markdown("---")
section("Financial Profile")
col4, col5, col6 = st.columns([1, 1, 1])
with col4:
    credit_score = st.number_input("Credit Score", value=600.0, min_value=300.0, max_value=850.0)
with col5:
    balance = st.number_input("Balance ($)", value=60000.0, min_value=0.0)
with col6:
    estimated_salary = st.number_input("Estimated Salary ($)", value=50000.0, min_value=0.0)

st.markdown("---")
section("Behaviour & Account")
col7, col8, col9, col10 = st.columns([1, 1, 1, 1])
with col7:
    age = st.slider("Age", 18, 92, 40)
with col8:
    tenure = st.slider("Tenure (years)", 0, 10, 3)
with col9:
    num_of_products = st.slider("Number of Products", 1, 4, 2)
with col10:
    is_active_member = st.selectbox("Is Active Member", [1, 0], format_func=lambda x: "Yes" if x else "No")

st.markdown("---")

# ── Metrics strip ─────────────────────────────────────────────────────────────
m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Age",        age)
m2.metric("Balance",    f"${balance:,.0f}")
m3.metric("Products",   num_of_products)
m4.metric("Tenure",     f"{tenure} yr")
m5.metric("Credit",     int(credit_score))
m6.metric("Salary",     f"${estimated_salary:,.0f}")

st.markdown("<br>", unsafe_allow_html=True)

# ── Predict button ────────────────────────────────────────────────────────────
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
        accent  = "#e24b4a"
        bg      = "rgba(226,75,74,0.07)"
        border  = "rgba(226,75,74,0.4)"
        divider = "rgba(226,75,74,0.15)"
        icon    = "&#9888;"
        verdict = "LIKELY TO CHURN"
        note    = "Consider a targeted retention offer — loyalty discount, personal outreach, or product review."
    else:
        accent  = "#ffffff"
        bg      = "rgba(255,255,255,0.04)"
        border  = "rgba(255,255,255,0.2)"
        divider = "rgba(255,255,255,0.08)"
        icon    = "&#10003;"
        verdict = "LIKELY TO STAY"
        note    = "Customer profile is stable. Continue standard engagement and monitor periodically."

    header_html = (
        '<div style="display:flex;align-items:center;gap:16px;margin-bottom:16px;flex-wrap:wrap;">'
        f'<span style="font-size:30px;color:{accent};">{icon}</span>'
        '<div style="flex:1;min-width:160px;">'
        '<div style="font-family:Share Tech Mono,monospace;font-size:10px;'
        'letter-spacing:1.5px;color:#444444;margin-bottom:4px;">PREDICTION OUTPUT</div>'
        f'<div style="font-family:Rajdhani,sans-serif;font-size:22px;font-weight:700;'
        f'color:{accent};letter-spacing:2px;">{verdict}</div>'
        '</div>'
        '<div style="text-align:right;">'
        '<div style="font-family:Share Tech Mono,monospace;font-size:10px;'
        'letter-spacing:1.5px;color:#444444;margin-bottom:2px;">CHURN PROBABILITY</div>'
        f'<div style="font-family:Share Tech Mono,monospace;font-size:36px;font-weight:700;color:{accent};">'
        f'{pct}%</div>'
        '</div></div>'
    )

    bar_html = (
        '<div style="background:#1a1a1a;border-radius:4px;height:6px;overflow:hidden;margin-bottom:6px;">'
        f'<div style="width:{pct}%;height:100%;background:{accent};border-radius:4px;"></div>'
        '</div>'
        '<div style="display:flex;justify-content:space-between;'
        'font-family:Share Tech Mono,monospace;font-size:10px;color:#444444;">'
        '<span>0%</span><span>50%</span><span>100%</span>'
        '</div>'
    )

    note_html = (
        f'<div style="margin-top:16px;padding-top:14px;border-top:1px solid {divider};'
        'font-family:Share Tech Mono,monospace;font-size:12px;color:#666666;line-height:1.7;">'
        f'&gt; {note}'
        '</div>'
    )

    card_html = (
        f'<div style="background:{bg};border:1px solid {border};border-radius:12px;'
        f'padding:24px 28px;margin-top:8px;">'
        + header_html + bar_html + note_html
        + '</div>'
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(card_html, unsafe_allow_html=True)