# 🧠 Mental Health Treatment Prediction App

A Machine Learning-powered web application built using **Streamlit** that predicts whether a person is likely to seek mental health treatment based on various factors.

---

# Live Demo
https://bhardwaj-komal-mental-health-prediction-app-app-nhojad.streamlit.app/
📸 App Preview
<img width="832" height="712" alt="image" src="https://github.com/user-attachments/assets/df17050b-4ccf-49e9-904a-fd1735f18bd5" />





## 🚀 Features

* Predict mental health treatment likelihood
* User-friendly web interface using Streamlit
* Machine Learning model using Random Forest
* Real-time predictions
* Clean and simple UI

---

## 🛠️ Tech Stack

* Python
* Streamlit
* Pandas
* NumPy
* Scikit-learn

---

## 📂 Project Structure

```
mental-health-app/
│
├── app.py              # Streamlit UI
├── model.py            # Model training script
├── model.pkl           # Saved ML model
├── encoders.pkl        # Label encoders
├── survey.csv          # Dataset
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository

```
git clone https://github.com/YOUR-USERNAME/mental-health-prediction-app.git
cd mental-health-prediction-app
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Run the app

```
streamlit run app.py
```

---

## 📊 Machine Learning Model

* Algorithm: Random Forest Classifier
* Handles categorical data using Label Encoding
* Trained on mental health survey dataset

---

## 🎯 Input Features

* Age
* Gender
* Family History
* Work Interference
* Remote Work
* Tech Company
* Mental Health Consequence

---

## 📌 Output

* ✅ Likely to seek treatment
* ⚠️ Not likely to seek treatment

---

## 🌐 Deployment

You can deploy this app using **Streamlit Cloud**.

---

## 🤝 Contributing

Feel free to fork this repository and improve the project!
