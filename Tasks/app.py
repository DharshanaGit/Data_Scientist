import streamlit as st
import numpy as np
import joblib

# ---------------- LOAD MODEL & SCALER ----------------
model = joblib.load("/home/intellect/Documents/Data_Scientist/Tasks/Students_final_score.pkl")
scaler = joblib.load("/home/intellect/Documents/Data_Scientist/Tasks/scaler.pkl")

st.set_page_config(page_title="Student Final Score Predictor", layout="centered")

st.title("🎓 Student Final Score Prediction")
st.write("Predict a student's **Final Score** using academic and personal factors.")

# ---------------- INPUT SECTION ----------------
st.subheader("📌 Enter Student Details")

Previous_Sem_Score = st.number_input(
    "Previous Semester Score",
    min_value=0.0, max_value=100.0, value=75.0
)

Study_Hours_per_Week = st.slider(
    "Study Hours per Week",
    min_value=0, max_value=70, value=20
)

Attendance_Percentage = st.slider(
    "Attendance Percentage",
    min_value=0, max_value=100, value=85
)

Family_Income = st.number_input(
    "Family Income",
    min_value=0.0, value=50000.0
)

Teacher_feedback_en = st.selectbox(
    "Teacher Feedback",
    options=[0, 1],
    format_func=lambda x: "Positive" if x == 1 else "Negative"
)

Sleep_Hours = st.slider(
    "Sleep Hours per Day",
    min_value=0, max_value=12, value=7
)

Internet_Access_en = st.selectbox(
    "Internet Access",
    options=[0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes"
)
Motivation_Level = st.slider(
    "Motivation Level (1–10)",
    min_value=1, max_value=10, value=6
)

Peer_Influence = st.slider(
    "Peer Influence (1–10)",
    min_value=1, max_value=10, value=5
)

Tutoring_Classes_en = st.selectbox(
    "Attends Tutoring Classes",
    options=[0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes"
)

# ---------------- PREDICTION ----------------
if st.button("🔮 Predict Final Score"):
    
    input_data = np.array([[
        Previous_Sem_Score,
        Study_Hours_per_Week,
        Attendance_Percentage,
        Family_Income,
        Teacher_feedback_en,
        Sleep_Hours,
        Internet_Access_en,
        Motivation_Level,
        Peer_Influence,
        Tutoring_Classes_en
    ]])

    # Apply StandardScaler
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]

    st.success(f"🎯 **Predicted Final Score: {prediction:.2f}**")

    # Interpretation
    if prediction >= 85:
        st.info("🌟 Excellent performance expected!")
    elif prediction >= 60:
        st.info("✅ Average to good performance expected.")
    else:
        st.warning("⚠️ Student may need academic support.")
