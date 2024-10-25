# app.py
import streamlit as st
import pandas as pd
import pickle

# Load the saved model
with open("loan_model.pkl", "rb") as f:
    model_lr = pickle.load(f)

# Define input fields directly in the app
st.title("Loan Approval Prediction")

# Sidebar for user input
st.sidebar.header("Input Loan Applicant Details")
def user_input_features():
    inputs = {
        "Gender": st.sidebar.selectbox("Gender", [0, 1]),  # 0 or 1, assuming binary
        "Married": st.sidebar.selectbox("Married", [0, 1]),  # 0 or 1
        "Dependents": st.sidebar.number_input("Dependents", min_value=0, max_value=3, step=1),  # Example range
        "Education": st.sidebar.selectbox("Education", [0, 1]),  # 0 or 1
        "Self_Employed": st.sidebar.selectbox("Self Employed", [0, 1]),  # 0 or 1
        "ApplicantIncome": st.sidebar.number_input("Applicant Income", min_value=0),
        "CoapplicantIncome": st.sidebar.number_input("Coapplicant Income", min_value=0),
        "LoanAmount": st.sidebar.number_input("Loan Amount", min_value=0),
        "Loan_Amount_Term": st.sidebar.number_input("Loan Amount Term", min_value=0),
        "Credit_History": st.sidebar.selectbox("Credit History", [0, 1]),  # 0 or 1
        "Property_Area": st.sidebar.selectbox("Property Area", [0, 1, 2])  # Assume 3 categories
    }
    return pd.DataFrame(inputs, index=[0])

# Capture user input data
user_inputs = user_input_features()

# Display input data
st.subheader("Applicant Details")
st.write(user_inputs)

# Prediction
if st.button("Predict"):
    prediction = model_lr.predict(user_inputs)[0]
    prediction_label = "Approved" if prediction == 1 else "Rejected"
    st.subheader("Prediction")
    st.write(f"The loan is likely to be **{prediction_label}**.")
