# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Heart Disease Predictor", layout="centered")

st.title("Heart Disease Predictor")
st.write("Enter patient data and get a prediction from the trained model.")

# Load artifacts once
@st.cache_resource
def load_artifacts():
    scaler = joblib.load("scaler.joblib")
    model = joblib.load("best_model.joblib")
    feature_cols = joblib.load("feature_columns.joblib")
    return scaler, model, feature_cols

scaler, model, feature_cols = load_artifacts()

# Create input widgets based on feature_cols
st.sidebar.header("Patient features")
input_data = {}
# If you know ranges for numeric columns you can set min/max; adjust as needed.
for col in feature_cols:
    # very common columns in heart dataset: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal
    if col in ["age"]:
        input_data[col] = st.sidebar.number_input(col, min_value=1, max_value=120, value=50)
    elif col in ["trestbps", "chol", "thalach"]:
        input_data[col] = st.sidebar.number_input(col, min_value=0, max_value=1000, value=120)
    elif col in ["oldpeak"]:
        input_data[col] = st.sidebar.number_input(col, min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    elif col in ["sex", "fbs", "exang", "restecg", "slope", "ca", "cp", "thal"]:
        # treat as integer-coded categories
        input_data[col] = st.sidebar.number_input(col, min_value=0, max_value=10, value=0)
    else:
        # fallback
        input_data[col] = st.sidebar.number_input(col, value=0.0)

if st.button("Predict"):
    # Create a single-row DataFrame with correct column order and dtypes
    X_new = pd.DataFrame([input_data], columns=feature_cols)
    # Scale using saved scaler
    X_new_scaled = scaler.transform(X_new)
    # Get prediction and probability
    preds = model.predict(X_new_scaled)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_new_scaled)
        # probability of class 1 (heart disease)
        prob_disease = probs[0][1]
    else:
        # for models without predict_proba, use decision_function or fallback
        prob_disease = None

    label = "Heart Disease (1)" if preds[0] == 1 else "No Heart Disease (0)"
    st.subheader("Prediction")
    st.write(f"**{label}**")
    if prob_disease is not None:
        st.progress(float(prob_disease))
        st.write(f"Predicted probability of heart disease: **{prob_disease:.2%}**")
    else:
        st.write("Probability not available for this model.")

    st.markdown("---")
    st.write("You can change values on the left and predict again.")

# Optional: show model comparison snapshot or sample data
if st.checkbox("Show model summary (from training)"):
    st.write("**Note:** training-time metrics are shown for reference.")
    try:
        # assume a file `training_summary.csv` exists or show a short message
        summary = pd.read_csv("training_summary.csv")
        st.dataframe(summary)
    except Exception:
        st.write("Training summary not available. Please include training_summary.csv if desired.")
