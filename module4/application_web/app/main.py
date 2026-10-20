from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .model_store import (
    analyse_g1,
    catalogue,
    load_metadata,
    predict_g1,
    predict_g2,
    predict_g3,
    predict_g4,
    predict_g5,
    predict_g6,
)


ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"

app = FastAPI(
    title="Laboratoire IA — ProDigiT",
    description="Interface d’inférence des six travaux pratiques de la journée J4.",
    version="1.0.0",
)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


class ValuesRequest(BaseModel):
    values: dict[str, float | int]


class TextRequest(BaseModel):
    text: str


class TerritoryRequest(BaseModel):
    region: str
    milieu: str


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "models": 6}


@app.get("/api/models")
def models() -> list[dict]:
    return catalogue()


@app.get("/api/models/{model_id}")
def model(model_id: str) -> dict:
    if model_id not in {f"g{index}" for index in range(1, 7)}:
        raise HTTPException(404, "Modèle inconnu")
    return load_metadata(model_id)


def run(function, *args):
    try:
        return function(*args)
    except ValueError as error:
        raise HTTPException(422, str(error)) from error


@app.post("/api/g1/analyse")
def g1(request: TerritoryRequest) -> dict:
    return run(analyse_g1, request.region, request.milieu)


@app.post("/api/g1/predict")
def g1_predict(request: ValuesRequest) -> dict:
    return run(predict_g1, request.values)


@app.post("/api/g2/predict")
def g2(request: ValuesRequest) -> dict:
    return run(predict_g2, request.values)


@app.post("/api/g3/predict")
def g3(request: ValuesRequest) -> dict:
    return run(predict_g3, request.values)


@app.post("/api/g4/predict")
def g4(request: ValuesRequest) -> dict:
    return run(predict_g4, request.values)


@app.post("/api/g5/predict")
def g5(request: ValuesRequest) -> dict:
    return run(predict_g5, request.values)


@app.post("/api/g6/predict")
def g6(request: TextRequest) -> dict:
    return run(predict_g6, request.text)
