
from pathlib import Path

import joblib 
import pandas as pd 

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel,Field

app = FastAPI(
    title="Iris Flower Prediction API",
    description="Predict Iris flower species using machine learning.",
    version="1.0.0",
)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "logistic_regression.pkl"
WEB_DIR = Path(__file__).resolve().parent / "web"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"File not Found : {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

app.mount("/static", StaticFiles(directory=WEB_DIR / "static"), name="static")

FEATURES = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]

class IrisInput(BaseModel):
    sepal_length: float = Field(gt=0,le=10)
    sepal_width: float = Field(gt=0,le=10)
    petal_length: float = Field(gt=0,le=10)
    petal_width: float = Field(gt=0,le=10)

@app.get("/")
def home():
    return FileResponse(WEB_DIR / "index.html")

@app.get("/api/info")
def api_info():
    return {
        "message": "Welcome to the Iris Flower Prediction API.",
        "docs": "/docs",
        "predict_endpoint": "/predict",
    }

@app.get("/health")
def health():
    return {
        "status" : "healthy",
        "model_load" : True
    }

@app.post("/predict")
def predict_species(iris:IrisInput):
    input_data = pd.DataFrame([[
        iris.sepal_length,
        iris.sepal_width,
        iris.petal_length,
        iris.petal_width
    ]],
    columns=FEATURES    
    )

    try:
        prediciton =model.predict(input_data)
        return {
            "message" : "Prediction Result",
            "Result" : f"{prediciton[0]}",
            "model" : "LogisticRegression",
            "input" : iris.model_dump(),
        }
    except Exception as error:
        raise HTTPException(status_code=500,detail="Prediction Failed.",) from error


      