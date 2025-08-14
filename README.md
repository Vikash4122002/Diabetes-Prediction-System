# 🩺 Diabetes Prediction App

An end-to-end machine learning app that predicts diabetes risk from patient inputs, deployed with Streamlit.

## 🌐 Live Demo
[Launch App](https://cbengpyzsjtcr6dg6jep3u.streamlit.app/)

## ✨ Features
- Real-time probability-based risk prediction
- Clean two-column UI with clinical reference ranges
- Visual risk indicator (progress bar) + actionable recommendations
- Robust error handling & cached model loading

## 🧠 Model
- Trained on standard Pima-style features: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DPF, Age  
- Evaluated via ROC-AUC & accuracy; best model serialized with Pickle

## 🛠 Tech Stack
Python, NumPy, scikit-learn, Streamlit, Pillow (for icons/images), Pickle

## 🚀 Run Locally
```bash
git clone https://github.com/YOUR_USERNAME/diabetes-prediction-app.git
cd diabetes-prediction-app
pip install -r requirements.txt
streamlit run app.py   # or main.py
