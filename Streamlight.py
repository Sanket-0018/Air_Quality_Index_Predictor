import streamlit as st
import numpy as np
import pickle

# Load model and scaler
with open("knn_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Title
st.title("AQI Predictor 🌫️")
st.markdown("Enter today's air quality readings to predict **tomorrow's AQI**.")

# Input fields
co = st.number_input("CO (Carbon Monoxide) in mg/m3", value=None, placeholder="Enter co value")
no2 = st.number_input("NO2 (Nitrogen Dioxide) in µg/m3",value=None, placeholder="Enter no2 value")
pm25 = st.number_input("PM2.5 (Fine Particulate Matter) in µg/m3", value=None, placeholder="Enter pm25 value")

# Predict button
if st.button("Predict AQI"):
    input_data = np.array([[co, no2, pm25]])
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)[0]
    prediction = float(prediction)
    prediction=round(prediction,2)

    # Classify AQI range
    if prediction <= 50:
        category = "Good 😌"
        color = "green"
    elif prediction <= 100:
        category = "Moderate 😐"
        color = "orange"
    else:
        category = "Poor 😷"
        color = "red"

    st.markdown(f"### Predicted AQI: `{prediction}`")
    st.markdown(f"### Category: <span style='color:{color}'>{category}</span>", unsafe_allow_html=True)
