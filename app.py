# app.py — Final AquaGuard (Purple result theme + logo + animations + footer)
import streamlit as st
import pandas as pd
from catboost import CatBoostClassifier
import time

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AquaGuard 💧", page_icon="💦", layout="wide")

# ---------------- CUSTOM CSS & HTML ----------------
css = r"""
<style>
/* FULL AQUA-GRADIENT SOFT BACKGROUND */
[data-testid="stAppViewContainer"] {
  background: linear-gradient(135deg,#e8f8ff,#d6fbff,#cdeffd,#e6e0ff);
  background-attachment: fixed;
  background-size: cover;
}

/* Hide default header */
[data-testid="stHeader"] { background: rgba(0,0,0,0); }

/* Top-left logo */
.logo {
  position: fixed;
  top: 18px;
  left: 22px;
  z-index: 9999;
  font-weight: 900;
  color: #2b2b2b;
  font-size: 20px;
  background: linear-gradient(90deg,#ffffffcc,#ffffff88);
  padding: 8px 12px;
  border-radius: 10px;
  border: 1px solid rgba(0,0,0,0.06);
  box-shadow: 0 6px 18px rgba(0,0,0,0.08);
}

/* HERO TITLE */
.hero-title {
  font-size: 46px;
  font-weight: 900;
  color: #071a2f;
  text-align: center;
  margin-top: 48px;
  text-shadow: 0 3px 12px rgba(123, 63, 183, 0.06);
}

/* GLASS PANEL */
.glass {
  background: rgba(255,255,255,0.62);
  border-radius: 16px;
  padding: 26px;
  backdrop-filter: blur(8px);
  border: 1px solid rgba(255,255,255,0.6);
  box-shadow: 0 10px 30px rgba(16,24,40,0.06);
}

/* INPUT labels & fields styling */
label, .css-16idsys p, .st-bc {
  color: #0b2540 !important;
  font-weight: 800 !important;
}
.stNumberInput input {
  background: rgba(255,255,255,0.95) !important;
  color: #001222 !important;
  font-weight: 800;
  border-radius: 10px;
}

/* Button style */
.stButton>button {
  background: linear-gradient(90deg,#7b3fb7,#a15be0);
  color: white;
  padding: 10px 22px;
  border-radius: 12px;
  font-weight: 800;
  border: none;
  transition: transform 0.18s ease, box-shadow 0.18s ease;
}
.stButton>button:hover {
  transform: translateY(-3px);
  box-shadow: 0 12px 28px rgba(120,50,160,0.28);
}

/* RESULT BANNERS - PURPLE THEME */
.result-safe {
  background: linear-gradient(90deg,#efe7ff,#f5ecff);
  border-left: 6px solid #8a4fe8;
  color: #2a154a;
  padding: 14px 18px;
  border-radius: 12px;
  font-weight: 900;
  box-shadow: 0 8px 20px rgba(120,60,180,0.06);
  margin-top: 14px;
}

.result-unsafe {
  background: linear-gradient(90deg,#fff0f0,#ffecec);
  border-left: 6px solid #d9534f;
  color: #5b1f1f;
  padding: 14px 18px;
  border-radius: 12px;
  font-weight: 900;
  box-shadow: 0 8px 20px rgba(200,40,40,0.04);
  margin-top: 14px;
}

/* Loader text */
.loading {
  font-weight: 800;
  color: #2b1b3a;
}

/* Animated water ripple footer */
.ripple-footer {
  position: fixed;
  left: 0;
  bottom: 0;
  width: 100%;
  height: 80px;
  background: linear-gradient(180deg, rgba(0,0,0,0), rgba(12,22,35,0.02));
  pointer-events: none;
  z-index: 999;
}
.wave {
  position: absolute;
  left: 0;
  bottom: 10px;
  width: 200%;
  height: 60px;
  background: rgba(138,79,232,0.12);
  filter: blur(12px);
  transform: translate3d(0,0,0);
  animation: waveMove 8s linear infinite;
  border-radius: 40%;
}
.wave2 {
  position: absolute;
  left: -50%;
  bottom: 6px;
  width: 200%;
  height: 50px;
  background: rgba(138,79,232,0.08);
  filter: blur(8px);
  animation: waveMove 10s linear infinite reverse;
  border-radius: 40%;
}
@keyframes waveMove {
  0% { transform: translateX(0); }
  100% { transform: translateX(-50%); }
}

/* small responsive tweaks */
@media (max-width: 900px) {
  .hero-title { font-size: 34px; margin-top: 20px; }
  .glass { padding: 18px; }
}
</style>

<!-- TOP-LEFT LOGO -->
<div class="logo">💧 <strong>AquaGuard</strong></div>

"""
st.markdown(css, unsafe_allow_html=True)

# ---------------- HERO ----------------
st.markdown("<div class='hero-title'>✨ AQUAGUARD – AI Powered Water Safety Checker ✨</div>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#0b2540; font-weight:700; margin-top:6px;'>Ensuring pure & safe water for every life — powered by smart ML.</p>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    m = CatBoostClassifier()
    m.load_model("catboost_water_model.cbm")
    return m

model = load_model()

# ---------------- GLASS PANEL (FORM) ----------------
st.markdown("<div class='glass'>", unsafe_allow_html=True)
st.markdown("<h3 style='color:#071a2f; font-weight:900;'>💧 Enter Water Parameters</h3>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
ph = col1.number_input("pH ", 0.0, 14.0, 7.0, format="%.2f")
hardness = col2.number_input("Hardness (mg/L)", 0.0, 500.0, 180.0, format="%.2f")
solids = col3.number_input("Solids (ppm)", 0.0, 50000.0, 15000.0, format="%.2f")

col4, col5, col6 = st.columns(3)
chloramines = col4.number_input("Chloramines (ppm)", 0.0, 15.0, 7.5, format="%.2f")
sulfate = col5.number_input("Sulfate (mg/L)", 0.0, 500.0, 330.0, format="%.2f")
conductivity = col6.number_input("Conductivity (µS/cm)", 0.0, 1000.0, 500.0, format="%.2f")

col7, col8, col9 = st.columns(3)
organic_carbon = col7.number_input("Organic Carbon (mg/L)", 0.0, 50.0, 10.0, format="%.2f")
trihalomethanes = col8.number_input("Trihalomethanes (µg/L)", 0.0, 150.0, 70.0, format="%.2f")
turbidity = col9.number_input("Turbidity (NTU)", 0.0, 10.0, 3.0, format="%.2f")

# ---------------- PREDICT BUTTON ----------------
if st.button("🔍 Predict Potability"):

    df_input = pd.DataFrame([[ph, hardness, solids, chloramines, sulfate, conductivity,
                              organic_carbon, trihalomethanes, turbidity]],
                            columns=["ph","Hardness","Solids","Chloramines","Sulfate",
                                     "Conductivity","Organic_carbon","Trihalomethanes","Turbidity"])

    # show a spinner + custom loading text
    with st.spinner("🔬 Analyzing water quality with AquaGuard..."):
        time.sleep(1.0)  # small pause so users see animation (feel free to adjust)
        pred = model.predict(df_input)[0]

    # manual safe checks as extra rule
    safe_cond = (
        (6.5 <= ph <= 8.5) and (120 <= hardness <= 220) and (5000 <= solids <= 25000) and
        (6 <= chloramines <= 9) and (250 <= sulfate <= 400) and (400 <= conductivity <= 700) and
        (8 <= organic_carbon <= 15) and (50 <= trihalomethanes <= 90) and (2 <= turbidity <= 4)
    )

    # Show purple-themed result banners with clear readable text
    if safe_cond or pred == 1:
        st.markdown("<div class='result-safe'>💜 <span style='font-size:18px'>The water is <strong>SAFE for Drinking</strong> — AquaGuard indicates good potability.</span></div>", unsafe_allow_html=True)
        st.markdown("<div style='margin-top:8px;color:#2b154a;font-weight:700;'>Suggestion: Continue regular monitoring. Consider filtration if taste/odor issues are observed.</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='result-unsafe'>⛔ <span style='font-size:18px'>The water is <strong>NOT SAFE</strong> for direct drinking. Purification recommended.</span></div>", unsafe_allow_html=True)
        st.markdown("<div style='margin-top:8px;color:#5b1f1f;font-weight:700;'>Suggestion: Use boiling/filtration/RO or contact local treatment before consumption.</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- CSV BATCH PREDICTION ----------------
st.markdown("<br><div class='glass'>", unsafe_allow_html=True)
st.markdown("### 📂 Upload CSV for Batch Prediction", unsafe_allow_html=True)

file = st.file_uploader("Upload CSV (columns: ph, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity):", type=["csv"])
if file:
    try:
        df = pd.read_csv(file)
        df = df.fillna(df.median())
        cols = ["ph","Hardness","Solids","Chloramines","Sulfate","Conductivity","Organic_carbon","Trihalomethanes","Turbidity"]
        if not all(c in df.columns for c in cols):
            st.error("❌ CSV missing required columns.")
        else:
            with st.spinner("🔬 Batch analyzing..."):
                time.sleep(0.8)
                preds = model.predict(df[cols])
            df["Prediction"] = ["Safe" if p==1 else "Not Safe" for p in preds]
            st.success("✅ Batch predictions complete!")
            st.dataframe(df)
            csv = df.to_csv(index=False).encode()
            st.download_button("⬇️ Download Results (CSV)", csv, "aquaguard_predictions.csv", "text/csv")
    except Exception as e:
        st.error(f"Error processing CSV: {e}")

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- FOOTER (RIPPLE) ----------------
footer_html = """
<div class="ripple-footer">
  <div class="wave"></div>
  <div class="wave2"></div>
</div>
"""
st.markdown(footer_html, unsafe_allow_html=True)

