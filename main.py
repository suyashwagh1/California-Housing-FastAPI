from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="California Housing Price Prediction API")

# Load model
model = joblib.load("california_housing_model.joblib")


class HouseFeatures(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float

    class Config:
        json_schema_extra = {
            "example": {
                "MedInc": 8.3252,
                "HouseAge": 41.0,
                "AveRooms": 6.9841,
                "AveBedrms": 1.0238,
                "Population": 322.0
            }
        }


@app.get("/")
def read_root():
    return {"message": "Welcome to the Housing Price Prediction API!"}


@app.post("/predict")
def predict_price(features: HouseFeatures):
    input_data = pd.DataFrame([features.dict()])
    prediction = model.predict(input_data)
    return {"predicted_median_house_value": float(prediction[0])}