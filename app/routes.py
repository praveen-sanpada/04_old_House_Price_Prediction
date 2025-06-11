# app/routes.py
from flask import Blueprint, render_template, request
import pickle
import numpy as np

main = Blueprint('main', __name__)

# Load models
with open('app/models/linear_regression_model.pkl', 'rb') as f:
    linear_model = pickle.load(f)
with open('app/models/polynomial_regression_model.pkl', 'rb') as f:
    poly_model = pickle.load(f)

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    final_features = np.array(features).reshape(1, -1)
    linear_prediction = linear_model.predict(final_features)[0]
    poly_prediction = poly_model.predict(final_features)[0]
    return render_template('result.html', linear_pred=linear_prediction, poly_pred=poly_prediction)
