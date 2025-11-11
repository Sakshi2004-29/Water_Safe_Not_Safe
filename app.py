import streamlit as st
import pandas as pd
from catboost import CatBoostClassifier

# ---------------------- PAGE SETUP ----------------------
st.set_page_config(page_title="💧 Water Potability Checker", page_icon="💦", layout="centered")

st.title("💧 Water Potability Prediction App")
st.write("Check if water is **Safe for Drinking** or **Not Safe** using your test values or CSV file.")

# ---------------------- LOAD TRAINED MODEL ----------------------
@st.cache_resource
def load_model():
    model = CatBoostClassifier()
    model.load_model("catboost_water_model.cbm")
    return model

model = load_model()

# ---------------------- SINGLE INPUT ----------------------
st.header("🧪 Enter Water Parameters")
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

if st.button("🔍 Predict Potability"):
    input_data = pd.DataFrame([[ph, hardness, solids, chloramines, sulfate, conductivity,
                                organic_carbon, trihalomethanes, turbidity]],
                              columns=["ph", "Hardness", "Solids", "Chloramines",
                                       "Sulfate", "Conductivity", "Organic_carbon",
                                       "Trihalomethanes", "Turbidity"])
    prediction = model.predict(input_data)[0]

    # Manual logical check for safe range
# inside your button handler, after you create input_data and prediction
# ------------------- SINGLE PREDICTION (with manual override) -------------------
# prediction = model.predict(input_data)[0]   # make sure you have this line above

# Manual logical check for safe range
if (6.5 <= ph <= 8.5) and (120 <= hardness <= 220) and (5000 <= solids <= 25000) and \
   (6 <= chloramines <= 9) and (250 <= sulfate <= 400) and (400 <= conductivity <= 700) and \
   (8 <= organic_carbon <= 15) and (50 <= trihalomethanes <= 90) and (2 <= turbidity <= 4):
    st.success("✅ The water is **SAFE for Drinking.** 💧 (Based on ideal parameter range)")
else:
    if prediction == 1:
        st.success("✅ The water is **SAFE for Drinking.** 💧")
    else:
        st.error("🚫 The water is **NOT SAFE for Drinking.** Please purify before use.")

# ---------------------- BATCH PREDICTION ----------------------
st.markdown("---")
st.header("📂 Upload CSV for Batch Prediction")
file = st.file_uploader("Upload CSV file with same column names as dataset", type=["csv"])

if file is not None:
    try:
        df = pd.read_csv(file)
        # handle missing values just in case
        df = df.fillna(df.median())

        # ensure columns order/names match model expectation
        expected_cols = ["ph", "Hardness", "Solids", "Chloramines",
                         "Sulfate", "Conductivity", "Organic_carbon",
                         "Trihalomethanes", "Turbidity"]
        if not all(col in df.columns for col in expected_cols):
            st.error(f"CSV must contain columns: {expected_cols}")
        else:
            preds = model.predict(df[expected_cols])
            df["Prediction"] = ["Safe" if p == 1 else "Not Safe" for p in preds]

            st.success("✅ Predictions Complete!")
            st.dataframe(df)
            csv = df.to_csv(index=False).encode()
            st.download_button("⬇️ Download Predictions CSV", csv, "predictions.csv", "text/csv")

    except Exception as e:
        st.error(f"⚠️ Error while processing file: {e}")





