import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="Europe Hotel Booking Satisfaction Predictor", layout="centered")
st.title("Europe Hotel Booking Satisfaction Predictor")

MODEL_PATH = Path("models/hotel_satisfaction_pipeline.joblib")
if not MODEL_PATH.exists():
    st.error("Model file not found at models/hotel_satisfaction_pipeline.joblib. Please upload it.")
    st.stop()

pipe = joblib.load(MODEL_PATH)

# These columns MUST match the training script's feature_cols exactly
feature_cols = [
    "Gender",
    "Age",
    "purpose_of_travel",
    "Type of Travel",
    "Type Of Booking",
    "Hotel wifi service",
    "Hotel location",
    "Food and drink",
    "Stay comfort",
    "Cleanliness",
]

with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.slider("Age", 18, 90, 30)
        purpose = st.selectbox("Purpose of Travel", ["Business", "Leisure"])
        travel_type = st.selectbox("Type of Travel", ["Solo", "Group"])
        booking = st.selectbox("Type Of Booking", ["Online", "Offline"])

    with col2:
        wifi = st.slider("Hotel wifi service", 1, 5, 3)
        location = st.slider("Hotel location", 1, 5, 3)
        food = st.slider("Food and drink", 1, 5, 3)
        comfort = st.slider("Stay comfort", 1, 5, 3)
        cleanliness = st.slider("Cleanliness", 1, 5, 3)

    submitted = st.form_submit_button("Predict Satisfaction")

if submitted:
    # Create a single-row DataFrame with the exact columns
    row = {
        "Gender": gender,
        "Age": age,
        "purpose_of_travel": purpose,
        "Type of Travel": travel_type,
        "Type Of Booking": booking,
        "Hotel wifi service": wifi,
        "Hotel location": location,
        "Food and drink": food,
        "Stay comfort": comfort,
        "Cleanliness": cleanliness,
    }
    X = pd.DataFrame([row], columns=feature_cols)

    # Predict
    pred = pipe.predict(X)[0]
    proba = getattr(pipe, "predict_proba", None)
    if proba is not None:
        probs = pipe.predict_proba(X)[0]
        # Show top class with probability
        if isinstance(pred, str):
            label = pred
        else:
            label = str(pred)
        st.success(f"Prediction: {label}")
        # Optional details
        st.write("Class probabilities (sorted):")
        # Build a small table
        prob_df = pd.DataFrame({"class": pipe.classes_, "probability": probs})
        prob_df = prob_df.sort_values("probability", ascending=False).reset_index(drop=True)
        st.dataframe(prob_df, use_container_width=True)
    else:
        st.success(f"Prediction: {pred}")
