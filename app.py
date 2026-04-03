import streamlit as st
import pandas as pd
import pickle

# Load model & encoders
model = pickle.load(open("model.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))

st.title("🧠 Mental Health Treatment Prediction")

st.write("Fill the details below:")

# User Inputs
age = st.number_input("Age", min_value=10, max_value=100, value=25)

gender = st.selectbox("Gender", ["Male", "Female", "Other"])
family_history = st.selectbox("Family History", ["Yes", "No"])
work_interfere = st.selectbox("Work Interfere", ["Often", "Sometimes", "Rarely", "Never"])
remote_work = st.selectbox("Remote Work", ["Yes", "No"])
tech_company = st.selectbox("Tech Company", ["Yes", "No"])
mental_health_consequence = st.selectbox("Mental Health Consequence", ["Yes", "No", "Maybe"])

if st.button("Predict"):

    input_data = pd.DataFrame([[ 
        age, gender, family_history,
        work_interfere, remote_work,
        tech_company, mental_health_consequence
    ]], columns=[
        "Age", "Gender", "family_history",
        "work_interfere", "remote_work",
        "tech_company", "mental_health_consequence"
    ])

    # Encode
    for col in input_data.columns:
        if col in encoders:
            try:
                input_data[col] = encoders[col].transform(input_data[col])
            except:
                st.error(f"Invalid input for {col}")
                st.stop()

    # Prediction
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("💡 Person is likely to seek treatment.")
    else:
        st.warning("⚠️ Person is NOT likely to seek treatment.")