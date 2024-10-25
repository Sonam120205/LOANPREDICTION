# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load data
data = pd.read_csv("LoanApprovalPrediction.csv")
data.drop(['Loan_ID'], axis=1, inplace=True)

# Preprocess data
X = data.drop(['Loan_Status'], axis=1)
Y = data['Loan_Status']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=1)

# Train the model
model_lr = LogisticRegression()
model_lr.fit(X_train, Y_train)

# Streamlit app
st.title("Loan Approval Prediction")

# Sidebar for user input
st.sidebar.header("Input Loan Applicant Details")
def user_input_features():
    inputs = {}
    for col in X.columns:
        # Assuming numerical input for simplicity; can be extended for categorical data
        inputs[col] = st.sidebar.number_input(f"{col}", float(X[col].min()), float(X[col].max()))
    return pd.DataFrame(inputs, index=[0])

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
