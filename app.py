from flask import Flask, render_template, request
import pandas as pd
from joblib import load

app = Flask(__name__)

# Load model, scaler, encoders
model = load('salary_prediction_model.joblib')
scaler = load('scaler.joblib')
label_encoders = load('label_encoders.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_data = {
        'age': [int(request.form['age'])],
        'workclass': [request.form['workclass']],
        'fnlwgt': [int(request.form['fnlwgt'])],
        'education': [request.form['education']],
        'educational-num': [int(request.form['educational-num'])],
        'marital-status': [request.form['marital-status']],
        'occupation': [request.form['occupation']],
        'relationship': [request.form['relationship']],
        'race': [request.form['race']],
        'gender': [request.form['gender']],
        'capital-gain': [int(request.form['capital-gain'])],
        'capital-loss': [int(request.form['capital-loss'])],
        'hours-per-week': [int(request.form['hours-per-week'])],
        'native-country': [request.form['native-country']]
    }

    df_input = pd.DataFrame(input_data)

    for col in df_input.select_dtypes(include='object').columns:
        df_input[col] = label_encoders[col].transform(df_input[col])

    df_scaled = scaler.transform(df_input)
    prediction = model.predict(df_scaled)
    result = ">50K" if prediction[0] == 1 else "<=50K"

    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
