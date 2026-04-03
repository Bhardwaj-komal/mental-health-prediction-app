
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load Dataset
dataset = pd.read_csv("survey.csv")

# Drop Unnecessary Columns
dataset = dataset.drop(columns=[
    'Timestamp', 'Country', 'state', 'self_employed',
    'no_employees',  'benefits',
    'care_options', 'wellness_program', 'seek_help',
    'anonymity', 'leave', 'phys_health_consequence',
    'coworkers', 'supervisor', 'mental_health_interview',
    'phys_health_interview', 'mental_vs_physical',
    'obs_consequence', 'comments'
], errors='ignore')


# Handle Missing Values
dataset = dataset.dropna()

# Encode Categorical Columns
encoders = {}
categorical_cols = dataset.select_dtypes(include=['object', 'string']).columns

for col in categorical_cols:
    le = LabelEncoder()
    dataset[col] = le.fit_transform(dataset[col])
    encoders[col] = le   # store encoder for future use

# Separate Features & Target
X = dataset.drop("treatment", axis=1)
y = dataset["treatment"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=0
)
# Train Model
# classifier = RandomForestClassifier(random_state=0)
# classifier.fit(X_train, y_train)

classifier = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,                 # limit tree depth
    min_samples_split=10,        # require more samples to split
    min_samples_leaf=5,          # prevent tiny leaf nodes
    random_state=0
)
classifier.fit(X_train, y_train)

# Accuracy Check
from sklearn.metrics import accuracy_score
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Overall Model Accuracy:", round(accuracy * 100, 2), "%")

# USER INPUT SECTION
# print("\nEnter Details for Prediction:")
# age = int(input("Enter Age: "))
# gender = input("Enter Gender (Male/Female/Other): ").strip().title()
# family_history = input("Family History (Yes/No): ").strip().title()
# work_interfere = input("Work Interfere (Often/Sometimes/Rarely/Never): ").strip().title()
# remote_work = input("Remote Work (Yes/No): ").strip().title()
# tech_company = input("Do you work in a Tech Company? (Yes/No): ").strip().title()
# mental_health_consequence = input("Mental Health Consequence (Yes/No/Maybe): ").strip().title()

import streamlit as st
import pandas as pd

st.title("🧠 Mental Health Treatment Prediction")

st.write("Enter your details below:")

# User Inputs
age = st.number_input("Age", min_value=10, max_value=100, step=1)
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
family_history = st.selectbox("Family History", ["Yes", "No"])
work_interfere = st.selectbox("Work Interfere", ["Often", "Sometimes", "Rarely", "Never"])
remote_work = st.selectbox("Remote Work", ["Yes", "No"])
tech_company = st.selectbox("Tech Company", ["Yes", "No"])
mental_health_consequence = st.selectbox("Mental Health Consequence", ["Yes", "No", "Maybe"])

if st.button("Predict"):
    input_data = pd.DataFrame([[ 
        age, gender, family_history,
        work_interfere, remote_work, tech_company,
        mental_health_consequence
    ]], columns=X.columns)

    # Encode input
    for col in input_data.columns:
        if col in encoders:
            try:
                input_data[col] = encoders[col].transform(input_data[col])
            except:
                st.error(f"Invalid value for {col}")

    prediction = classifier.predict(input_data)

    if prediction[0] == 1:
        st.success("Person is likely to seek treatment")
    else:
        st.warning("Person is NOT likely to seek treatment")

# Create DataFrame in same column order as X
input_data = pd.DataFrame([[ 
    age, gender, family_history,
    work_interfere, remote_work,tech_company,
    mental_health_consequence
]], columns=X.columns)

# Encode Input Using Saved Encoders
for col in input_data.columns:
    if col in encoders:
        try:
            input_data[col] = encoders[col].transform(input_data[col])
        except ValueError:
            print(f"Error: Invalid value entered for {col}")
            exit()

# Make Prediction
prediction = classifier.predict(input_data)

if prediction[0] == 1:
    print("\nPerson is likely to seek treatment.")
else:
    print("\nPerson is NOT likely to seek treatment.")


# #  Accuracy Check
# from sklearn.metrics import accuracy_score
# y_pred = classifier.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print("Overall Model Accuracy:", round(accuracy * 100, 2), "%")