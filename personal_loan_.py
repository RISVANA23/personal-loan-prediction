import streamlit as st
import joblib
import numpy as np


st.title("Personal Loan Prediction")


model = joblib.load("personal_loan.pkl")
scaler = joblib.load("scaler.pkl")

customer_id = st.number_input("Customer ID", min_value=1, step=1)
age = st.number_input("Age", min_value=18, max_value=100, value=35)

monthly_income = st.number_input(
    "Monthly Income", min_value=0, max_value=100000, value=50000)

monthly_expenses = st.number_input(
    "Monthly Expenses", min_value=0, max_value=100000, value=20000)

monthly_savings = st.number_input(
    "Monthly Savings", min_value=0, max_value=100000, value=10000)

credit_score = st.number_input(
    "Credit Score", min_value=300, max_value=900, value=700)

loan_amount = st.number_input(
    "Loan Amount", min_value=0, max_value=5000000, value=500000)

risk_level = st.selectbox(
    "Risk Level", ["Low", "Medium", "High"])





if st.button("Predict"):

    risk_map = {
        "Low": 0,
        "Medium": 1,
        "High": 2
    }

    risk_level_encoded = risk_map[risk_level]

    data = np.array([[customer_id,age,
                      monthly_income,
                      monthly_expenses,
                      monthly_savings,
                      credit_score,
                      loan_amount,
                      risk_level_encoded]])

    data = scaler.transform(data)

    prediction = model.predict(data)[0]

    if prediction == 1:
        st.success("Customer WILL take Personal Loan")
    else:
        st.info("Customer will NOT take Personal Loan")
