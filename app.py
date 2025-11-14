import streamlit as st
import pandas as pd
from catboost import CatBoostClassifier

# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(
    page_title="💧 Water Potability Checker",
    page_icon="💦",
    layout="wide"
)

# ---------------------- PREMIUM BACKGROUND + GLASS CSS ----------------------
css = """
<style>

/* FULL WATER BACKGROUND IMAGE */
[data-testid="stAppViewContainer"] {
    background-image: url("https://images.unsplash.com/photo-1508685096489-7aacd43bd3b1?auto=format&fit=crop&w=1920&q=80");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

/* Remove white header */
[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}

/* HERO TEXT STYLE */
.hero-title {
    font-size: 55px;
    font-weight: 800;
    color: #ffffff;
    text-shadow: 0px 0px 20px rgba(0,255,255,0.7);
    text-align: center;
    margin-top: 40px;
}

.hero-sub {
    font-size: 20px;
    text-align: center;
    color: #e3faff;
    margin-top: -15px;
}

/* GLASS CONTAINER */
.glass-box {
    background: rgba(255, 255, 255, 0.15);
    border-radius: 18px;
    padding: 30px;
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.25);
    box-shadow: 0 0 25px rgba(0,0,0,0.2);
}

/* BUTTON STYLE */
.stButton>button {
    background: linear-gradient(45deg, #00d4ff, #0077b6);
    padding: 10px 25px;
    color: white;
    border-radius: 12px;
    border: none;
    font-size: 18px;
    transition: 0.3s;
}

.stButton>button:hover {
    transform: scale(1.05);
    box-shadow: 0 0 20px #00eaff;
}

/* FIELDS */
.stNumberInput input {
    background: rgba(255,255,255,0.20) !important;
    color: white !important;
    border-radius: 10px;
}

label, .css-16idsys p {
    color: #eaffff !important;
    font-weight: 600;
}

</style>
"""
st.markdown(css, unsafe_allow_html=True)

# ---------------------- HERO SECTION ----------------------
st.markdown("<h1 class='hero-title'>Always Want Safe Water<br>For Healthy Life</h1>", unsafe_allow_html=True)
st.markdown("<p class='hero-sub'>Check water potability using AI — ensure purity before drinking.</p>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ---------------------- GLASS CARD START ----------------------
st.markdown("<div class='glass-box'>", unsafe_allow_html=True)

st.markdown("### 💧 Enter Water Parameters")

# ---------------------- LOAD MODEL ----------------------
@st.cache_resource
def load_model():
    model = CatBoostClassifier()
    model.load_model("catboost_water_model.cbm")
    return model

model = load_model()

# ---------------------- INPUT FIELDS ----------------------
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

# ---------------------- GLASS DIV END ----------------------
st.markdown("</div>", unsafe_allow_html=True)

# ---------------------- CSV SECTION ----------------------
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
        st.error("❌ Incorrect column names!")

st.markdown("</div>", unsafe_allow_html=True)
