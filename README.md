# SafeSip – AI Based Water Potability Prediction System

<br>

## 📌 Project Overview

This project predicts whether water is safe for drinking based on chemical and physical water quality parameters.
Using advanced Machine Learning models and a Streamlit-based UI, the system helps detect water safety quickly and accurately.

<br> 

## 🎯 Problem Statement

Many regions do not have access to instant water quality testing tools.
Manual lab testing is time-consuming and expensive, which leads to health risks.

This project solves the problem by:

✦ Predicting potability using ML

✦ Providing instant results

✦ Helping people avoid unsafe drinking water

<br> 

## 📂 Dataset

The dataset contains 3276 rows with 9 water quality parameters:

Feature	Description
pH	Acidity/Alkalinity
Hardness	Mineral concentration
Solids	Total dissolved solids
Chloramines	Water disinfectant level
Sulfate	Sulfur content
Conductivity	Electrical conductivity
Organic Carbon	Carbon concentration
Trihalomethanes	Disinfection by-products
Turbidity	Water clarity
Potability	Safe (1) / Unsafe (0)
🔧 Preprocessing Steps

✔ Handling missing values
✔ Distribution analysis (histograms)
✔ Correlation heatmap
✔ Standard scaling
✔ Train-test splitting
✔ Outlier analysis

🤖 Models Used
Model	Tuned With	Accuracy
Random Forest	Optuna	XX%
AdaBoost	Optuna	XX%
CatBoost	Optuna	XX%
XGBoost	Optuna	XX%
LightGBM	Optuna	XX%

(XX म्हणजे तुझी actual accuracy values टाक)

🧪 Model Evaluation

The project evaluates models using:

Accuracy

Precision

Recall

F1-score

Confusion Matrix

ROC Curve (optional)

🎨 Web App UI (Streamlit)

Features:
✔ Aqua gradient background
✔ Glass effect input box
✔ Glow effects
✔ Logo + animations
✔ “Safe / Not Safe” prediction banner
✔ Manual WHO rule safety + ML prediction
✔ CSV batch prediction
✔ Downloadable results file

🚀 How to Run Locally
1️⃣ Clone the Repository
git clone https://github.com/yourusername/aquaguard.git
cd aquaguard

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Run Streamlit App
streamlit run app.py

🧪 Tech Stack

Python

Pandas, NumPy

Scikit-Learn

CatBoost, LightGBM, XGBoost

Optuna (Hyperparameter Tuning)

Streamlit

Matplotlib, Seaborn

🛠 Project Structure
aquaguard/
│── app.py               # Streamlit Web App  
│── model/               # Saved Models  
│── data/                # Dataset  
│── notebooks/           # EDA, preprocessing  
│── README.md            # Project documentation  
│── requirements.txt     # Dependency list  

🌎 Real-World Applications

Drinking water quality testing

Rural water supply monitoring

IoT-enabled water purification systems

Government water safety dashboard

Smart city water management

🧑‍🎓 Team Members

Sakshi Patil

Others (if group)

🏁 Conclusion

AquaGuard successfully predicts water potability using optimized machine learning models.
This project demonstrates ML + MLOps concepts and real-life application potential.
