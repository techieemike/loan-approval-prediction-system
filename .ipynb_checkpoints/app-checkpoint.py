import streamlit as st
import joblib
import pandas as pd
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))
from src.feature_engineering import preprocess_features_for_input
import feature_engineering as fe
from src import feature_engineering as fe

# app.py


# Load model
model = joblib.load("model/best_loan_model.pkl")

# Title
st.title("Loan Default Prediction App")
st.markdown("Enter applicant details below to predict the likelihood of loan default.")

# Input form
with st.form("prediction_form"):
    Gender = st.selectbox("Gender", ["Male", "Female"])
    Married = st.selectbox("Married", ["Yes", "No"])
    Dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    Education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    Self_Employed = st.selectbox("Self Employed", ["Yes", "No"])
    ApplicantIncome = st.number_input("Applicant Income", min_value=0)
    CoapplicantIncome = st.number_input("Coapplicant Income", min_value=0)
    LoanAmount = st.number_input("Loan Amount", min_value=0)
    Loan_Amount_Term = st.number_input("Loan Amount Term", min_value=0)
    Credit_History = st.selectbox("Credit History", [1.0, 0.0])
    Property_Area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
    
    submitted = st.form_submit_button("Predict")

# Predict
if submitted:
    input_data = {
        'Gender': Gender,
        'Married': Married,
        'Dependents': Dependents,
        'Education': Education,
        'Self_Employed': Self_Employed,
        'ApplicantIncome': ApplicantIncome,
        'CoapplicantIncome': CoapplicantIncome,
        'LoanAmount': LoanAmount,
        'Loan_Amount_Term': Loan_Amount_Term,
        'Credit_History': Credit_History,
        'Property_Area': Property_Area
    }

    try:
        X_processed = fe.preprocess_features_for_input(input_data)
        prediction = model.predict(X_processed)[0]
        prediction_proba = model.predict_proba(X_processed)[0][1]  # Probability of default

        st.success(f"Prediction: {'Loan Will Default' if prediction == 1 else 'Loan Will Not Default'}")
        st.info(f"Default Probability: {prediction_proba:.2f}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
