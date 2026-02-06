# California Housing Price Prediction API (FastAPI + Machine Learning)

This project trains a **RandomForestRegressor** model using the **California Housing dataset** and deploys it as a **REST API** using **FastAPI**.

The API accepts housing-related features as input and returns the predicted median house value.

---

## 🚀 Features
- Train a Machine Learning model using Scikit-learn
- Save trained model using Joblib
- Build REST API using FastAPI
- Predict median house values through `/predict` endpoint
- Auto-generated API documentation using Swagger UI (`/docs`)

---

## 🛠️ Tech Stack
- Python
- Pandas
- Scikit-learn
- Joblib
- FastAPI
- Uvicorn

---

## 📂 Project Structure
california-housing-fastapi/
│── train.py
│── main.py
│── requirements.txt
│── README.md
│── .gitignore
│── california_housing_model.joblib