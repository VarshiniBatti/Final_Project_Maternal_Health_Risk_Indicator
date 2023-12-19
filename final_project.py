from flask import Flask, render_template, request
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb


app = Flask(__name__)

cleaned_final_data = pd.read_csv('data_cleaned.csv')

#Pickle file 
with open('xgboost_model.pkl', 'rb') as model_file:
    xgb_model = pickle.load(model_file)

# loading label encoder
with open('label_encoder_xgb.pkl', 'rb') as le_file:
    label_enc_xgb = pickle.load(le_file)

# Define acceptable ranges for input values
valid_ranges = {
    'age': (10, 100),
    'systolic_bp': (60, 190),
    'diastolic_bp': (30, 150),
    'heart_rate': (40, 150),
    'bs': (2, 20),
}

# for the homepage 
@app.route('/')
def index():
    return render_template('index.html')

# for the predictor page 
@app.route('/predictor')
def predictor():
    return render_template('predictor.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get values from physiological features
        systolic_bp = float(request.form['systolic_bp'])
        diastolic_bp = float(request.form['diastolic_bp'])
        heart_rate = float(request.form['heart_rate'])
        age = float(request.form['age'])
        bs = float(request.form['bs'])

        # Validate input values
        validation_errors = []
        for name, value in {'age': age, 'systolic_bp': systolic_bp, 'diastolic_bp': diastolic_bp, 'heart_rate': heart_rate, 'bs': bs}.items():
            min_val, max_val = valid_ranges[name]
            if not min_val <= value <= max_val:
                validation_errors.append(f"{name.capitalize()} should be between {min_val} and {max_val}.")

        # If there are errors in values, run error.html
        if validation_errors:
            return render_template('error.html', error_messages=validation_errors)

        # Create a dataframe with the values given by user 
        input_data = pd.DataFrame([[systolic_bp, diastolic_bp, heart_rate, age, bs]],
                                  columns=['SystolicBP', 'DiastolicBP', 'HeartRate', 'Age', 'BS'])   

        # Predict with model 
        prediction = xgb_model.predict(input_data)
        risk_level_xgb = label_enc_xgb.inverse_transform(prediction)[0]

        # Give the predictions and show them on result.html
        return render_template('result.html', risk_level=risk_level_xgb) 
    
# for visualizations page 
@app.route('/visualizations')
def visualizations():
    return render_template('visualizations.html')
    
# for the contact page 
@app.route('/contact_us')
def contact_us():
    return render_template('contact_us.html')

if __name__ == "__main__":
    app.run(debug=True)