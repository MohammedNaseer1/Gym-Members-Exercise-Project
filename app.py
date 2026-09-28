from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).parent / "models" / "calories_prediction_model.pkl"

st.set_page_config(page_title="Calories Burned Predictor", page_icon="🏋️")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()

st.title("Gym Calories Burned Predictor")
st.write(
    "Estimate the calories burned in one gym session. "
    "The model is a Linear Regression trained on a gym members dataset."
)

st.subheader("Member profile")
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 18, 59, 35)
weight = st.slider("Weight (kg)", 40.0, 130.0, 70.0, step=0.5)
height = st.slider("Height (m)", 1.50, 2.00, 1.70, step=0.01)
fat_percentage = st.slider("Body fat (%)", 10.0, 35.0, 25.0, step=0.5)
water_intake = st.slider("Daily water intake (liters)", 1.5, 3.7, 2.6, step=0.1)

st.subheader("Workout details")
workout_type = st.selectbox("Workout type", ["Cardio", "HIIT", "Strength", "Yoga"])
session_duration = st.slider("Session duration (hours)", 0.5, 2.0, 1.0, step=0.05)
workout_frequency = st.slider("Workout frequency (days/week)", 2, 5, 3)
experience_level = st.selectbox("Experience level", [1, 2, 3])

st.subheader("Heart rate (BPM)")
max_bpm = st.slider("Max BPM", 160, 199, 180)
avg_bpm = st.slider("Average BPM", 120, 169, 145)
resting_bpm = st.slider("Resting BPM", 50, 74, 62)

if st.button("Predict calories burned"):
    member = pd.DataFrame(
        [
            {
                "Age": age,
                "Weight (kg)": weight,
                "Height (m)": height,
                "Max_BPM": max_bpm,
                "Avg_BPM": avg_bpm,
                "Resting_BPM": resting_bpm,
                "Session_Duration (hours)": session_duration,
                "Fat_Percentage": fat_percentage,
                "Water_Intake (liters)": water_intake,
                "Workout_Frequency (days/week)": workout_frequency,
                "Experience_Level": experience_level,
                "Gender_Male": int(gender == "Male"),
                "Workout_Type_HIIT": int(workout_type == "HIIT"),
                "Workout_Type_Strength": int(workout_type == "Strength"),
                "Workout_Type_Yoga": int(workout_type == "Yoga"),
            }
        ]
    )
    member = member[list(model.feature_names_in_)]

    prediction = max(float(model.predict(member)[0]), 0.0)
    st.metric("Estimated calories burned", f"{prediction:,.0f} kcal")
    st.caption(
        "On the test set, the model's average error was about 30 kcal. "
        "The inputs are limited to the ranges found in the training data."
    )

st.divider()
st.caption(
    "Portfolio project. The dataset appears to be synthetic, so this is a demo, "
    "not medical or training advice."
)