
import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the trained model, scaler, and label encoder
try:
    model = joblib.load('random_forest_classification_model.pkl')
    scaler = joblib.load('standard_scaler.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    feature_names = joblib.load('feature_names.pkl')
except FileNotFoundError:
    st.error("Model, scaler, label encoder or feature names files not found. "
             "Please ensure 'random_forest_classification_model.pkl', 'standard_scaler.pkl', 'label_encoder.pkl', "
             "and 'feature_names.pkl' are in the same directory.")
    st.stop()

st.title("Risk Classification Prediction App")
st.write("Enter the feature values below to predict the risk classification.")

# Create input fields for each feature dynamically
input_data = {}
for feature in feature_names:
    input_data[feature] = st.number_input(f"Enter {feature}", value=0.0) # Default to 0.0, adjust as needed

# Convert input data to DataFrame
input_df = pd.DataFrame([input_data])

if st.button("Predict Risk Classification"):
    # Scale the input features
    scaled_input = scaler.transform(input_df)

    # Make prediction
    prediction_numeric = model.predict(scaled_input)

    # Inverse transform the prediction to get original label
    prediction_label = label_encoder.inverse_transform(prediction_numeric)

    st.success(f"Predicted Risk Classification: {prediction_label[0]}")

st.markdown("---")
st.markdown("This app uses a trained Random Forest Classifier to predict risk.")
