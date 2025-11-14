import streamlit as st
import pandas as pd
from catboost import CatBoostClassifier

# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(
    page_title="💧 Water Potability Checker",
    page_icon="💦",
    layout="wide"
)

# ---------------------- CUSTOM CSS (Aqua Glassmorphism) ----------------------
page_bg = """
<style>
/* Background Gradient */
body {
    background: linear-gradient(135deg, #0d1b2a, #1d3557, #0a4f70);
    color: white;
}

/* Glassmorphism Card */
.reportview-container .main .block-container {
    backdrop-filter: blur(15px);
    background: rgba(255, 255, 255, 0.08);
    padding: 2rem;
    border-radius: 20px;
    border: 1px solid rgba(255,255,255,0.2);
    box-shadow: 0 8px 32px rgba(0,0,0,0.2);
}

/* Headers */
h1, h2, h3 {
    font-weight: 700 !important;
    text-shadow: 0px 0px 10px #00c8ff;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(45deg, #00c8ff, #0077b6);
    color: white;
    padding: 0.7rem 1.2rem;
    border-radius: 10px;
    border: none;
    font-size: 18px;
    transition: 0.3s;
}
.stButton>button:hover {
    transform: scale(1.05);
    box-shadow: 0 0 15px #00eaff;
}

/* Input Fields */
.css-q8sbsg, .stNumberInput input {
    background-color: rgba(255,255,255,0.15) !important;
    color: white !important;
}

/* Divider */
hr {
    border: 1px solid rgba(255,255,255,0.3);
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# ---------------------- TITLE ----------------------
st.markdown("<h1 style='text-align:center;'>💧 Water Potability Prediction App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Check if water is <b>Safe</b> or <b>Not Safe</b> using test values or CSV.</p>", unsafe_allow_html=True)

# ---------------------- LOAD MODEL ----------------------
@st.cache_resource
def load_model():
    model = CatBoostClassifier()
    model.load_model("catboost_water_model.cbm")
    return model

model = load_model()

st.markdown("### 🧪 Enter Water Parameters")

# ---------------------- INPUTS ----------------------
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

# ---------------------- SINGLE PREDICTION ----------------------
if st.button("🔍 Predict Potability"):
    input_data = pd.DataFrame([[ph, hardness, solids, chloramines, sulfate, conductivity,
                                organic_carbon, trihalomethanes, turbidity]],
                              columns=["ph", "Hardness", "Solids", "Chloramines",
                                       "Sulfate", "Conductivity", "Organic_carbon",
                                       "Trihalomethanes", "Turbidity"])
    
    prediction = model.predict(input_data)[0]

    # Manual logical check for safe range
    safe_cond = (
        (6.5 <= ph <= 8.5) and (120 <= hardness <= 220) and (5000 <= solids <= 25000) and
        (6 <= chloramines <= 9) and (250 <= sulfate <= 400) and (400 <= conductivity <= 700) and
        (8 <= organic_carbon <= 15) and (50 <= trihalomethanes <= 90) and (2 <= turbidity <= 4)
    )

    if safe_cond or prediction == 1:
        st.success("💧 The water is SAFE for Drinking.")
    else:
        st.error("🚫 The water is NOT SAFE. Please purify before use.")

# ---------------------- CSV UPLOAD ----------------------
st.markdown("---")
st.markdown("### 📂 Upload CSV for Batch Prediction")

file = st.file_uploader("Upload CSV file with same column names:", type=["csv"])

if file is not None:
    try:
        df = pd.read_csv(file).fillna(df.median())

        expected_cols = ["ph", "Hardness", "Solids", "Chloramines",
                         "Sulfate", "Conductivity", "Organic_carbon",
                         "Trihalomethanes", "Turbidity"]

        if not all(col in df.columns for col in expected_cols):
            st.error(f"CSV must contain columns: {expected_cols}")
        else:
            preds = model.predict(df[expected_cols])
            df["Prediction"] = ["Safe" if p == 1 else "Not Safe" for p in preds]

            st.success("✅ CSV Predictions Complete!")
            st.dataframe(df)

            csv = df.to_csv(index=False).encode()
            st.download_button("⬇️ Download Results CSV", csv, "water_predictions.csv", "text/csv")

    except Exception as e:
        st.error(f"⚠️ Error: {e}")
