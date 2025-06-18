# ❤️ Heart Disease Prediction using Machine Learning

This project predicts the presence of heart disease using multiple machine learning models. It includes a user-friendly web interface built with Flask, allowing real-time predictions based on user input.

---

## 🚀 Features

- Accepts patient details via web form
- Predicts heart disease risk using ML models
- Compares performance across 7 algorithms
- Displays colored output: green (safe), red (at risk)
- Clean UI with Flask + HTML + CSS

---

## 🧠 Models Implemented & Accuracy

| Model                    | Accuracy (%) |
|-------------------------|--------------|
| Logistic Regression     | 86.96        |
| Naive Bayes             | 82.61        |
| Support Vector Machine  | 85.33        |
| K-Nearest Neighbors     | 70.65        |
| Decision Tree           | 88.04        |
| Random Forest           | **90.22**    |
| Neural Network (MLP)    | 86.41        |

> ✅ **Highest Accuracy**: Random Forest (90.22%)

---

## 📊 Dataset

This project uses the [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+Disease) which includes features like:

- Age
- Gender
- Chest Pain Type
- Resting Blood Pressure
- Cholesterol
- Fasting Blood Sugar
- ECG Results
- Max Heart Rate
- Exercise Induced Angina
- ST Depression
- Slope of ST Segment
- Major Vessels (CA)
- Thalassemia

---

## 🛠️ Tech Stack

- **Backend**: Python, Flask
- **Frontend**: HTML, CSS, Jinja2
- **Machine Learning**: scikit-learn 
- **Model Persistence**: joblib
- **Data Handling**: pandas, numpy

---

