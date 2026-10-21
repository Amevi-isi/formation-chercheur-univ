from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"


@lru_cache(maxsize=8)
def load_model(model_id: str) -> dict:
    return joblib.load(MODELS / f"{model_id}.joblib")


@lru_cache(maxsize=8)
def load_metadata(model_id: str) -> dict:
    return json.loads((MODELS / f"{model_id}.json").read_text(encoding="utf-8"))


def catalogue() -> list[dict]:
    return json.loads((MODELS / "catalogue.json").read_text(encoding="utf-8"))


def validate_numbers(model_id: str, values: dict) -> tuple[pd.DataFrame, list[str]]:
    metadata = load_metadata(model_id)
    fields = metadata.get("fields", [])
    row, warnings = {}, []
    for field in fields:
        name = field["name"]
        if name not in values:
            raise ValueError(f"Champ manquant : {name}")
        try:
            value = float(values[name])
        except (TypeError, ValueError) as error:
            raise ValueError(f"Valeur non numérique : {name}") from error
        if not np.isfinite(value):
            raise ValueError(f"Valeur non finie : {name}")
        if field["type"] == "boolean" and value not in (0, 1):
            raise ValueError(f"{field['label']} doit valoir 0 ou 1")
        if value < field["min"] or value > field["max"]:
            warnings.append(f"{field['label']} est hors de la plage centrale des données d’apprentissage.")
        row[name] = value
    return pd.DataFrame([row]), warnings


def analyse_g1(region: str, milieu: str) -> dict:
    metadata = load_metadata("g1")
    rows = [item for item in metadata["summaries"] if item["region"] == region and item["milieu"] == milieu]
    if not rows:
        raise ValueError("Cette combinaison région–milieu n’existe pas dans le jeu de données.")
    return rows[0]


def predict_g1(values: dict) -> dict:
    payload = load_model("g1")
    frame, warnings = validate_numbers("g1", values)
    frame = frame[payload["features"]]
    probability = float(payload["model"].predict_proba(frame)[0, 1])
    return {
        "prediction": "Manque de soins prédit" if probability >= 0.5 else "Absence de manque de soins prédite",
        "probability": probability,
        "threshold": 0.5,
        "warnings": warnings,
    }


def predict_g2(values: dict) -> dict:
    payload = load_model("g2")
    frame, warnings = validate_numbers("g2", values)
    frame = frame[payload["features"]]
    probabilities = payload["model"].predict_proba(frame)[0]
    order = np.argsort(probabilities)[::-1]
    classes = payload["model"].classes_
    return {
        "prediction": str(classes[order[0]]),
        "probabilities": [{"label": str(classes[i]), "value": float(probabilities[i])} for i in order],
        "warnings": warnings,
    }


def predict_g3(values: dict) -> dict:
    payload = load_model("g3")
    metadata = load_metadata("g3")
    frame, warnings = validate_numbers("g3", values)
    prediction = float(payload["model"].predict(frame[payload["features"]])[0])
    interval = metadata["interval90"]
    return {
        "prediction": prediction,
        "unit": "°C",
        "interval": [prediction - interval, prediction + interval],
        "warnings": warnings,
    }


def predict_g4(values: dict) -> dict:
    payload = load_model("g4")
    frame, warnings = validate_numbers("g4", values)
    frame = frame[payload["features"]]
    probability = float(payload["model"].predict_proba(frame)[0, 1])
    return {
        "prediction": "Profil proche du groupe MRC" if probability >= 0.5 else "Profil proche du groupe sans MRC",
        "probability": probability,
        "warnings": warnings,
    }


def predict_g5(values: dict) -> dict:
    payload = load_model("g5")
    metadata = load_metadata("g5")
    frame, warnings = validate_numbers("g5", values)
    frame["log_revenu"] = np.log(frame["pib_par_habitant"])
    prediction = float(payload["model"].predict(frame[payload["features"]])[0])
    interval = metadata["interval90"]
    return {
        "prediction": prediction,
        "unit": "ans",
        "interval": [prediction - interval, prediction + interval],
        "warnings": warnings,
    }


def predict_g6(text: str) -> dict:
    if len(text.split()) < 25:
        raise ValueError("Le passage doit contenir au moins 25 mots pour obtenir un résultat interprétable.")
    payload = load_model("g6")
    pipeline = payload["model"]
    probabilities = pipeline.predict_proba([text])[0]
    classifier = pipeline.named_steps["logisticregression"]
    vectorizer = pipeline.named_steps["tfidfvectorizer"]
    order = np.argsort(probabilities)[::-1]

    predicted_index = int(order[0])
    vector = vectorizer.transform([text])
    contributions = vector.multiply(classifier.coef_[predicted_index]).toarray()[0]
    vocabulary = vectorizer.get_feature_names_out()
    best_words = [vocabulary[i] for i in np.argsort(contributions)[::-1] if contributions[i] > 0][:8]
    return {
        "prediction": str(classifier.classes_[predicted_index]),
        "probabilities": [
            {"label": str(classifier.classes_[i]), "value": float(probabilities[i])} for i in order
        ],
        "indicators": best_words,
        "word_count": len(text.split()),
        "warnings": [] if len(text.split()) >= 100 else ["Un passage plus long rend généralement l’attribution plus stable."],
    }
