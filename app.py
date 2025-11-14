import streamlit as st
import pandas as pd
from catboost import CatBoostClassifier

# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(
    page_title="AquaGuard 💧",
    page_icon="💦",
    layout="wide"
)

# ---------------------- AQUA GRADIENT + GLASS CSS ----------------------
css = """
<style>

/* FULL BACKGROUND */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg,#dff9fb,#c7ecee,#a5d8ff,#74b9ff,#81ecec);
    background-attachment: fixed;
    background-size: cover;
}

/* TRANSPARENT HEADER */
[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}

/* HERO TITLE */
.hero-title {
    font-size: 55px;
    font-weight: 900;
    color: #012a4a;
    text-align: center;
    margin-top: 40px;
    text-shadow: 0px 0px 8px rgba(0,100,255,0.15);
}

/* HERO SUBTITLE */
.hero-sub {
    font-size: 20px;
    font-weight: 600;
    text-align: center;
    color: #013a63;
    margin-top: -10px;
}

/* GLASS EFFECT MAIN CARD */
.glass-box {
    background: rgba(255,255,255,0.45);
    border-radius: 18px;
    padding: 30px;
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.4);
    box-shadow: 0 0 25px rgba(0,0,0,0.15);
}

/* BLACK INPUT LABELS */
label {
    color: #001f33 !important;
    font-weight: 700 !important;
}

/* INPUT FIELD */
.stNumberInput input {
    background: rgba(255,255,255,0.75) !important;
    color: #001f33 !important;
    font-weight: 700;
    border-radius: 10px;
}

/* BUTTON STYLE */
.stButton>button {
    background: linear-gradient(45deg,#0096c7,#0077b6);
    padding: 10px 25px;
    color: white;
    border-radius: 12px;
    border: none;
    font-size: 18px;
    font-weight: 700;
    transition: 0.3s;
}

.stButton>button:hover {
    transform: scale(1.07);
    box-shadow: 0 0 20px #00b4d8;
}

/* CSV Upload Text */
.css-16idsys p {
    color: #001f33 !important;
    font-weight: 700;
}

</style>
"""
st.markdown(css, unsafe_allow_html=True)

# ---------------------- HERO SECTION ----------------------
st.markdown("<h1 class='hero-title'>✨ AQUAGUARD – AI Powered Water Safety Checker ✨</h1>", unsafe_allow_html=True)
st.markdown("<p class='hero-sub'>Ensuring pure & safe water for every life — powered by Smart Machine Learning.</p>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)


# ---------------------- LOAD MODEL ----------------------
@st.cache_resource
def load_model():
    model = CatBoostClassifier()
    model.load_model("catboost_water_model.cbm")
    return model

model = load_model()

# ---------------------- GLASS PANEL ----------------------
st.markdown("<div class='glass-box'>", unsafe_allow_html=True)

st.markdown("### 💧 Enter Water Parameters")

col1, col2, col3 = st.columns(3)
ph = col1.number_input("pH (6.5–8.5)", 0.0, 14.0, 7.0)
hardness = col2.number_input("Hardness (mg/L)", 0.0, 500.0, 180.0)
solids = col3.number_input("Solids (ppm)", 0.0, 50000.0, 15000.0)

col4, col5, col6 = st.columns(3)
chloramines = col4.number_input("Chloramines (ppm)", 0.0, 15.0, 7.5)
sulfate = col5.number_input("Sulfate (mg/L)", 0.0, 500.0, 330.0)
conductivity = col6.number_input("Conductivity (µS/cm)", 0.0, 1000.0, 500.0)

col7, col8, col9 = st.columns(3)
organic_carbon = col7.number_input("Organic Carbon (mg/L)", 0.0, 50.0, 10.0)
trihalomethanes = col8.number_input("Trihalomethanes (µg/L)", 0.0, 150.0, 70.0)
turbidity = col9.number_input("Turbidity (NTU)", 0.0, 10.0, 3.0)


# ---------------------- PREDICTION ----------------------
if st.button("🔍 Predict Potability"):
    data = pd.DataFrame([[ph, hardness, solids, chloramines, sulfate, conductivity,
                          organic_carbon, trihalomethanes, turbidity]],
                        columns=["ph", "Hardness", "Solids", "Chloramines",
                                 "Sulfate", "Conductivity", "Organic_carbon",
                                 "Trihalomethanes", "Turbidity"])

    prediction = model.predict(data)[0]

    safe_cond = (
        (6.5 <= ph <= 8.5) and (120 <= hardness <= 220) and (5000 <= solids <= 25000) and
        (6 <= chloramines <= 9) and (250 <= sulfate <= 400) and (400 <= conductivity <= 700) and
        (8 <= organic_carbon <= 15) and (50 <= trihalomethanes <= 90) and (2 <= turbidity <= 4)
    )

    if safe_cond or prediction == 1:
        st.success("💙 The water is SAFE for Drinking!")
    else:
        st.error("🚫 The water is NOT SAFE. Please purify before use.")

st.markdown("</div>", unsafe_allow_html=True)


# ---------------------- CSV BOX ----------------------
st.markdown("<br><div class='glass-box'>", unsafe_allow_html=True)
st.markdown("### 📂 Upload CSV for Batch Prediction")

file = st.file_uploader("Upload CSV file:", type=["csv"])

if file:
    df = pd.read_csv(file).fillna(df.median())

    cols = ["ph", "Hardness", "Solids", "Chloramines",
            "Sulfate", "Conductivity", "Organic_carbon",
            "Trihalomethanes", "Turbidity"]

    if all(c in df.columns for c in cols):
        preds = model.predict(df[cols])
        df["Prediction"] = ["Safe" if p == 1 else "Not Safe" for p in preds]
        st.dataframe(df)
    else:
        st.error("❌ Wrong column names in CSV!")

st.markdown("</div>", unsafe_allow_html=True)
