import streamlit as st
import numpy as np
import joblib as jb

# -----------------------------
# Load saved model & scaler
# -----------------------------
model=jb.load("/home/intellect/Documents/Data_Scientist/KNN-Project/Social_network-ads.pkl")
scaler=jb.load("/home/intellect/Documents/Data_Scientist/KNN-Project/scaler.pkl")

# -----------------------------
# Title
# -----------------------------
st.title("📱 Social Network Ads - KNN Prediction")
st.write("Predict whether a user will purchase or not")

# -----------------------------
# User Input
# -----------------------------
age = st.number_input("Age", min_value=18, max_value=100, value=30)
salary = st.number_input("Estimated Salary", min_value=10000, max_value=200000, value=87000)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict"):
    new_data = np.array([[age, salary]])
    new_data_scaled = scaler.transform(new_data)
    prediction = model.predict(new_data_scaled)

    if prediction[0] == 1:
        st.success("✅ User will PURCHASE")
    else:
        st.error("❌ User will NOT purchase")
