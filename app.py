import pickle
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="Student Grade Predictor")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

with open(BASE_DIR / "student_model.pkl", "rb") as f:
    saved_data = pickle.load(f)
    model = saved_data["model"]
    transformer = saved_data["transformer"]
    scaler = saved_data["scaler"]


class StudentFeatures(BaseModel):
    school: str
    sex: str
    age: int = Field(ge=15, le=22)
    address: str
    famsize: str
    Pstatus: str
    Medu: int = Field(ge=0, le=4)
    Fedu: int = Field(ge=0, le=4)
    Mjob: str
    Fjob: str
    reason: str
    guardian: str
    traveltime: int = Field(ge=1, le=4)
    studytime: int = Field(ge=1, le=4)
    failures: int = Field(ge=0, le=3)
    schoolsup: str
    famsup: str
    paid: str
    activities: str
    nursery: str
    higher: str
    internet: str
    romantic: str
    famrel: int = Field(ge=1, le=5)
    freetime: int = Field(ge=1, le=5)
    goout: int = Field(ge=1, le=5)
    Dalc: int = Field(ge=1, le=5)
    Walc: int = Field(ge=1, le=5)
    health: int = Field(ge=1, le=5)
    absences: int = Field(ge=0, le=93)
    G1: int = Field(ge=0, le=20)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.post("/predict")
async def predict(features: StudentFeatures):
    try:
        input_data = pd.DataFrame([features.model_dump()])
        input_transformed = transformer.transform(input_data)
        input_scaled = scaler.transform(input_transformed)
        predicted = int(model.predict(input_scaled)[0])
        passed = predicted == 1
        probability = None
        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(input_scaled)[0][predicted])
        return {
            "prediction": predicted,
            "passed": passed,
            "label": "Pass" if passed else "Fail",
            "probability": probability,
        }
    except Exception as exc:
        return JSONResponse(status_code=500, content={"detail": str(exc)})
