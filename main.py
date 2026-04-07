import streamlit as st
import pickle
import pandas as pd

# 1. Load trained model
with open("rf_cv_best.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🏦 Loan Approval Prediction App")

# 2. User Inputs
credit_history = st.selectbox("Credit History", [0, 1])
loan_amount = st.number_input("Loan Amount", min_value=50, max_value=1000, step=10)
applicant_income = st.number_input("Applicant Income", min_value=0, step=100)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0, step=100)
property_area_semiurban = st.selectbox("Property Area Semiurban", [0, 1])

# 3. Prepare input data
input_data = pd.DataFrame({
    'Credit_History': [credit_history],
    'LoanAmount': [loan_amount],
    'ApplicantIncome': [applicant_income],
    'CoapplicantIncome': [coapplicant_income],
    'Property_Area_Semiurban': [property_area_semiurban]
})

# ⭐ IMPORTANT FIX (ADD THIS)
try:
    input_data = input_data.reindex(columns=model.feature_names_in_, fill_value=0)
except:
    st.error("Model does not have feature names. Retrain model properly.")
    st.stop()

# 4. Prediction
if st.button("Predict Loan Approval"):
    prediction = model.predict(input_data)

    st.subheader("📊 Result:")
    if prediction[0] == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")