
from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import pickle

# Initialize the Flask application
app = Flask(__name__)

# Load the model (assuming a pickle model file was saved previously)
# model = pickle.load(open('loan_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')  # You can create an index.html for a simple form

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from the form
    data = request.form

    # Process data - assuming similar structure to what you have in the notebook
    # Convert input into a DataFrame
    input_data = pd.DataFrame({
        'Gender': [data['Gender']],
        'Married': [data['Married']],
        'Dependents': [data['Dependents']],
        'Education': [data['Education']],
        'Self_Employed': [data['Self_Employed']],
        'ApplicantIncome': [data['ApplicantIncome']],
        'CoapplicantIncome': [data['CoapplicantIncome']],
        'LoanAmount': [data['LoanAmount']],
        'Loan_Amount_Term': [data['Loan_Amount_Term']],
        'Credit_History': [data['Credit_History']],
        'Property_Area': [data['Property_Area']]
    })

    # Here you would apply any preprocessing steps (like encoding categorical data, handling NaNs)
    # And then call model.predict(input_data) to get the prediction
    # For now, let's just return the input as a test response

    return jsonify(input_data.to_dict())

if __name__ == "__main__":
    app.run(debug=True)
