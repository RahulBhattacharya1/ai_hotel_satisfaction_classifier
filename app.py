import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("models/hotel_satisfaction_model.pkl")

st.title("Europe Hotel Booking Satisfaction Predictor")

# Input form
gender = st.selectbox("Gender", ["Male","Female"])
age = st.slider("Age", 18, 90, 30)
purpose = st.selectbox("Purpose of Travel", ["Business","Leisure"])
travel_type = st.selectbox("Type of Travel", ["Solo","Group"])
booking = st.selectbox("Type of Booking", ["Online","Offline"])

wifi = st.slider("Hotel wifi service",1,5,3)
location = st.slider("Hotel location",1,5,3)
food = st.slider("Food and drink",1,5,3)
comfort = st.slider("Stay comfort",1,5,3)
cleanliness = st.slider("Cleanliness",1,5,3)

if st.button("Predict Satisfaction"):
    # Convert to DataFrame for prediction
    input_df = pd.DataFrame([[gender, age, purpose, travel_type, booking, wifi, location, food, comfort, cleanliness]],
                            columns=["Gender","Age","purpose_of_travel","Type of Travel","Type Of Booking",
                                     "Hotel wifi service","Hotel location","Food and drink","Stay comfort","Cleanliness"])
    
    # Dummy encoding (should match training)
    mapping = {"Male":0,"Female":1,"Business":0,"Leisure":1,"Solo":0,"Group":1,"Online":0,"Offline":1}
    for col in ["Gender","purpose_of_travel","Type of Travel","Type Of Booking"]:
        input_df[col] = input_df[col].map(mapping)
    
    pred = model.predict(input_df)[0]
    st.success("Prediction: " + ("Satisfied" if pred==1 else "Dissatisfied"))
