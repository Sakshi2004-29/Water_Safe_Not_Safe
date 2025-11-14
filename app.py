import streamlit as st
import pandas as pd
from catboost import CatBoostClassifier

# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(
    page_title="💧 Water Potability Checker",
    page_icon="💦",
    layout="wide"
)

# ---------------------- BACKGROUND + GLASS EFFECT CSS ----------------------
bg_css = """
<style>

/* FULL BACKGROUND */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0A4F70, #1D3557, #0F2027);
    background-size: cover;
    background-attachment: fixed;
}

/* HEADER TRANSPARENT */
[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}

/* SIDEBAR GLASS EFFECT */
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.07);
    backdrop-filter: blur(12px);
    border-right: 1px solid rgba(255,255,255,0.2);
}

/* MAIN CARD GLASS EFFECT */
.block-container {
    background: rgba(255, 255, 255, 0.08);
    padding: 2rem 2rem;
    margin-top: 2rem;
    border-radius: 18px;
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.2);
    box-shadow: 0 0 30px rgba(0,255,255,0.15);
}

/* TITLE GLOW */
h1, h2, h3 {
    color: white !important;
    text-shadow: 0px 0px 20px #00eaff;
    font-weight: 700;
}

/* BUTTON STYLING */
.stButton>button {
    background: linear-gradient(45deg, #00c8ff, #0077b6);
    padding: 10px 25px;
    color: white;
    border-radius: 12px;
    border: none;
    font-size: 18px;
    transition: 0.3s;
}

.stButton>button:hover {
    transform: scale(1.07);
    box-shadow: 0 0 20px #00eaff;
}

/* INPUT FIELDS */
.stNumberInput input {
    background: rgba(255,255,255,0.12) !important;
    color: white !important;
    border-radius: 10px;
}

/* SLIDERS + LABELS */
label, .st-af {
    color: #E8F9FD !important;
}

</style>
"""

st.markdown(bg_css, unsafe_allow_html=True)


# ---------------------- TITLE ----------------------
st.markdown("<h1 style='text-align:center;'>💧 Water Potability Prediction App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;font-size:18px;'>Predict whether water is <b>Drinkable</b> or <b>Not Safe</b> using AI</p>", unsafe_allow_html=True)


# ---------------------- LOAD CATBOOST MODEL ----------------------
@st.cache_resource
def load_model():
    model = CatBoostClassifier()
    model.load_model("catboost_water_model.cbm")
    return model

model = load_model()


# ---------------------- INPUT SECTION ----------------------
st.markdown("### 🧪 Enter Water Parameters Below")

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


# ---------------------- SINGLE PREDICTION BUTTON ----------------------
if st.button("🔍 Predict Potability"):
    
    input_data = pd.DataFrame([[ph, hardness, solids, chloramines, sulfate, conductivity,
                                organic_carbon, trihalomethanes, turbidity]],
                              columns=["ph", "Hardness", "Solids", "Chloramines",
                                       "Sulfate", "Conductivity", "Organic_carbon",
                                       "Trihalomethanes", "Turbidity"])

    prediction = model.predict(input_data)[0]

    # Manual safe limits check
    safe_cond = (
        (6.5 <= ph <= 8.5) and (120 <= hardness <= 220) and (5000 <= solids <= 25000) and
        (6 <= chloramines <= 9) and (250 <= sulfate <= 400) and (400 <= conductivity <= 700) and
        (8 <= organic_carbon <= 15) and (50 <= trihalomethanes <= 90) and (2 <= turbidity <= 4)
    )

    if safe_cond or prediction == 1:
        st.success("💧 The water is SAFE for Drinking! 💙")
    else:
        st.error("🚫 The water is NOT SAFE. Please purify before use.")


# ---------------------- CSV UPLOAD ----------------------
st.markdown("---")
st.markdown("### 📂 Upload CSV for Batch Prediction")

file = st.file_uploader("Upload CSV file (same columns as dataset)", type=["csv"])

if file is not None:
    try:
        df = pd.read_csv(file)
        df = df.fillna(df.median())

        expected_cols = ["ph", "Hardness", "Solids", "Chloramines",
                         "Sulfate", "Conductivity", "Organic_carbon",
                         "Trihalomethanes", "Turbidity"]

        if not all(c in df.columns for c in expected_cols):
            st.error("CSV missing required columns!")
        else:
            preds = model.predict(df[expected_cols])
            df["Prediction"] = ["Safe" if p == 1 else "Not Safe" for p in preds]

            st.success("✅ Predictions Generated!")
            st.dataframe(df)

            csv = df.to_csv(index=False).encode()
            st.download_button("⬇️ Download Predictions", csv, "predictions.csv", "text/csv")

    except Exception as e:
        st.error(f"⚠️ Error: {e}")
