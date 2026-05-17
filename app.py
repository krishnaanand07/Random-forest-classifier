import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load("random_forest_model.joblib")

scaler = joblib.load("scaler.joblib")

feature_columns = joblib.load("feature_columns.joblib")

st.title("Customer Churn Prediction")

gender = st.number_input("Gender")
SeniorCitizen = st.number_input("SeniorCitizen")
Partner = st.number_input("Partner")
Dependents = st.number_input("Dependents")
tenure = st.number_input("Tenure")
PhoneService = st.number_input("PhoneService")
PaperlessBilling = st.number_input("PaperlessBilling")
MonthlyCharges = st.number_input("MonthlyCharges")
TotalCharges = st.number_input("TotalCharges")

input_data = np.array([[
    gender,
    SeniorCitizen,
    Partner,
    Dependents,
    tenure,
    PhoneService,
    PaperlessBilling,
    MonthlyCharges,
    TotalCharges
]])

if st.button("Predict"):

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.error("Customer Will Churn")
    else:
        st.success("Customer Will Not Churn")