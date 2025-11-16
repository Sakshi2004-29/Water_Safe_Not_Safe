# SafeSip – AI Based Water Potability Prediction System

<br>

## 🎯 Problem Statement

Many regions do not have access to instant water quality testing tools.
Manual lab testing is time-consuming and expensive, which leads to health risks.

This project solves the problem by:

✦ Predicting potability using ML

✦ Providing instant results

✦ Helping people avoid unsafe drinking water

<br> 

## 📌 Overview

This project predicts whether drinking water is Safe (1) or Not Safe (0) using chemical and physical water quality parameters.

The repository includes:

● Complete Jupyter/Colab notebook

● Trained ML models (CatBoost, XGBoost, LGBM, RF, AdaBoost)

● Hyperparameter tuning scripts (Optuna)

● Streamlit Web App (app.py)

● Dataset (water_potability_final.csv)

● Model performance comparison

● EDA visualizations

● Project Report 

<br>


## 🚀 Live Demo
https://watersafenotsafe.streamlit.app/
<br>


## 📊 Dataset Description
| Feature             | Description                      |
| ------------------- | -------------------------------- |
| **pH**              | Acidity/Basicity of water        |
| **Hardness**        | Mineral concentration            |
| **Solids**          | Total dissolved solids (TDS)     |
| **Chloramines**     | Water disinfectant concentration |
| **Sulfate**         | Sulfur minerals                  |
| **Conductivity**    | Electrical conductivity          |
| **Organic Carbon**  | Organic impurities               |
| **Trihalomethanes** | Disinfection by-products         |
| **Turbidity**       | Clarity of water                 |
| **Potability**      | 1 = Drinkable, 0 = Not Drinkable |
<br>


## 🧪 Machine Learning Models Used

| Model                      | Accuracy | Notes            |
| -------------------------- | -------- | ---------------- |
| **Random Forest (Optuna)** | ~83%     | Strong baseline  |
| **AdaBoost (Optuna)**      | ~74%     | Weak performer   |
| **CatBoost (Optuna)**      | ~84%     | Best accuracy    |
| **LightGBM (Optuna)**      | ~81%     | Fast & efficient |
| **XGBoost (Optuna)**       | ~81.6%   | Good balance     |

✔ Selected Final Model: CatBoostClassifier

  ✦ Best accuracy

  ✦ Handles missing values

  ✦ Handles non-linear patterns

  ✦ Fast training
  
<br>


## 🧠 ML Pipeline

○ Data Cleaning

○ Missing value handling

○ Feature Scaling (StandardScaler)

○ Train-test split (80-20)

○ Model training

○ Optuna hyperparameter tuning

○ Evaluation (Accuracy, Precision, Recall, F1-score)

○ Confusion Matrix

○ Final model selection

○ Streamlit deployment

<br>

## 🛠️ Tech Stack

1) Python

2) Pandas, NumPy

3) Scikit-Learn

4) Optuna (Hyperparameter Tuning)

5) CatBoost, XGBoost, LightGBM

6) Streamlit

7) Seaborn, Matplotlib

<br>

## 🌍 Real-World Applications

💧 Rural drinking water testing

💧 Household water purifier quality checks

💧 Environmental monitoring systems

💧 IoT-based water quality alerts

💧 Government water supply assessment

<br>

## 🏁 Conclusion

AquaGuard provides a fast, efficient, and accurate way to determine water potability using machine learning.
The system can be integrated into smart cities, IoT devices, and water treatment facilities.

<br>
