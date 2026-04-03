import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load Dataset
dataset = pd.read_csv("survey.csv")

# Drop Unnecessary Columns
dataset = dataset.drop(columns=[
    'Timestamp', 'Country', 'state', 'self_employed',
    'no_employees', 'benefits', 'care_options',
    'wellness_program', 'seek_help', 'anonymity',
    'leave', 'phys_health_consequence', 'coworkers',
    'supervisor', 'mental_health_interview',
    'phys_health_interview', 'mental_vs_physical',
    'obs_consequence', 'comments'
], errors='ignore')

# Handle Missing Values
dataset = dataset.dropna()

# Encode Categorical Columns
encoders = {}
categorical_cols = dataset.select_dtypes(include=['object']).columns

for col in categorical_cols:
    le = LabelEncoder()
    dataset[col] = le.fit_transform(dataset[col])
    encoders[col] = le

# Features & Target
X = dataset.drop("treatment", axis=1)
y = dataset["treatment"]

# Train Model
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=0
)

model.fit(X, y)

# Save model & encoders
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(encoders, open("encoders.pkl", "wb"))

print("✅ Model & encoders saved successfully!")