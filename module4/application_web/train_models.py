from __future__ import annotations

import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, recall_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import make_pipeline


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)


def finite(value: float, digits: int = 3) -> float:
    return round(float(value), digits)


def numeric_schema(frame: pd.DataFrame, columns: list[str], labels: dict[str, str] | None = None) -> list[dict]:
    labels = labels or {}
    fields = []
    for column in columns:
        series = pd.to_numeric(frame[column], errors="coerce").dropna()
        fields.append(
            {
                "name": column,
                "label": labels.get(column, column.replace("_", " ").capitalize()),
                "type": "number",
                "default": finite(series.median()),
                "min": finite(series.quantile(0.01)),
                "max": finite(series.quantile(0.99)),
                "step": "any",
            }
        )
    return fields


def save(name: str, payload: object, metadata: dict) -> None:
    joblib.dump(payload, MODELS / f"{name}.joblib", compress=3)
    (MODELS / f"{name}.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def train_g1() -> dict:
    data = pd.read_csv(DATA / "afrobarometre_togo_2024.csv")
    data["cible"] = (data["manque_soins"] >= 2).astype(int)
    data["urbain"] = (data["milieu"] == "urbain").astype(int)
    data["femme"] = (data["sexe"] == "femme").astype(int)
    features = [
        "urbain", "age", "femme", "niveau_instruction", "electricite_zone",
        "eau_courante_zone", "egouts_zone", "forage_zone", "ecole_zone",
        "centre_sante_zone", "marche_zone", "banque_zone", "transport_zone",
        "menage_radio", "menage_television", "menage_ordinateur",
        "menage_compte_bancaire",
    ]
    clean = data.dropna(subset=features + ["manque_soins"]).copy()
    x_train, x_test, y_train, y_test = train_test_split(
        clean[features], clean["cible"], test_size=0.3, stratify=clean["cible"], random_state=42
    )
    evaluation = RandomForestClassifier(
        n_estimators=300, min_samples_leaf=5, random_state=42, n_jobs=-1
    ).fit(x_train, y_train)
    prediction = evaluation.predict(x_test)
    final = RandomForestClassifier(
        n_estimators=300, min_samples_leaf=5, random_state=42, n_jobs=-1
    ).fit(clean[features], clean["cible"])
    clean["probabilite_modele"] = final.predict_proba(clean[features])[:, 1]

    labels = {
        "urbain": "Milieu de résidence",
        "age": "Âge",
        "femme": "Sexe",
        "niveau_instruction": "Niveau d’instruction (0 à 9)",
        "electricite_zone": "Électricité présente dans la zone",
        "eau_courante_zone": "Eau courante présente dans la zone",
        "egouts_zone": "Égouts présents dans la zone",
        "forage_zone": "Forage présent dans la zone",
        "ecole_zone": "École présente dans la zone",
        "centre_sante_zone": "Centre de santé présent dans la zone",
        "marche_zone": "Marché présent dans la zone",
        "banque_zone": "Banque présente dans la zone",
        "transport_zone": "Transport disponible dans la zone",
        "menage_radio": "Radio dans le ménage",
        "menage_television": "Télévision dans le ménage",
        "menage_ordinateur": "Ordinateur dans le ménage",
        "menage_compte_bancaire": "Compte bancaire dans le ménage",
    }
    fields = numeric_schema(clean, features, labels)
    binary = set(features) - {"age", "niveau_instruction"}
    for field in fields:
        if field["name"] in binary:
            field.update({"type": "boolean", "default": int(round(field["default"])), "min": 0, "max": 1})
    for field in fields:
        if field["name"] == "urbain":
            field.update({"type": "choice", "options": [{"value": 0, "label": "Rural"}, {"value": 1, "label": "Urbain"}]})
        if field["name"] == "femme":
            field.update({"type": "choice", "options": [{"value": 0, "label": "Homme"}, {"value": 1, "label": "Femme"}]})

    summaries = []
    for (region, milieu), group in clean.groupby(["region", "milieu"]):
        summaries.append(
            {
                "region": str(region),
                "milieu": str(milieu),
                "n": int(len(group)),
                "observed_rate": finite(group["cible"].mean(), 4),
                "model_rate": finite(group["probabilite_modele"].mean(), 4),
            }
        )
    metadata = {
        "id": "g1",
        "title": "Territoire et accès aux soins",
        "question": "Quelle probabilité de manque de soins le modèle associe-t-il à un profil complet ?",
        "method": "Forêt aléatoire utilisant les 17 variables effectivement retenues dans le notebook.",
        "metrics": {
            "Exactitude sur le jeu test": finite(accuracy_score(y_test, prediction), 4),
            "Rappel — manque de soins": finite(recall_score(y_test, prediction), 4),
        },
        "regions": sorted(clean["region"].astype(str).unique().tolist()),
        "milieux": sorted(clean["milieu"].astype(str).unique().tolist()),
        "summaries": summaries,
        "fields": fields,
        "example": {name: finite(clean[name].median()) for name in features},
        "n": int(len(clean)),
        "scope": "Inférence sur des profils décrits par les mêmes 17 variables que les 1 072 observations exploitables de l’enquête.",
        "warning": "La probabilité produite décrit le comportement du modèle. Elle ne mesure pas une causalité et ne doit pas servir seule à décider pour une personne.",
    }
    save("g1", {"model": final, "features": features}, metadata)
    return metadata


def train_g2() -> dict:
    data = pd.read_csv(DATA / "haricots.csv")
    features = [column for column in data.columns if column != "variete"]
    x_train, x_test, y_train, y_test = train_test_split(
        data[features], data["variete"], test_size=0.3, stratify=data["variete"], random_state=42
    )
    evaluation = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1).fit(x_train, y_train)
    prediction = evaluation.predict(x_test)
    final = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1).fit(
        data[features], data["variete"]
    )
    examples = {}
    for variety, group in data.groupby("variete"):
        examples[str(variety)] = {name: finite(value) for name, value in group[features].median().items()}
    metadata = {
        "id": "g2",
        "title": "Caractérisation de variétés de haricots",
        "question": "À quelle variété un grain ressemble-t-il au regard de ses mesures morphologiques ?",
        "method": "Forêt aléatoire entraînée sur 16 mesures numériques.",
        "metrics": {"Exactitude sur le jeu test": finite(accuracy_score(y_test, prediction), 4)},
        "fields": numeric_schema(data, features),
        "examples": examples,
        "classes": [str(value) for value in final.classes_],
        "scope": "Classification morphométrique sur les variétés présentes dans le jeu d’apprentissage.",
        "warning": "Le modèle ne traite aucune image et ne peut reconnaître une variété absente de son apprentissage.",
    }
    save("g2", {"model": final, "features": features}, metadata)
    return metadata


def prepare_g3() -> pd.DataFrame:
    data = pd.read_csv(DATA / "meteo_lome.csv", parse_dates=["date"])
    data["tmax_hier"] = data["temperature_max"].shift(1)
    data["tmin_hier"] = data["temperature_min"].shift(1)
    data["humidite_hier"] = data["humidite"].shift(1)
    data["vent_hier"] = data["vent"].shift(1)
    data["ensoleillement_hier"] = data["ensoleillement"].shift(1)
    data["pluie_hier"] = data["pluie"].shift(1)
    data["mois"] = data["date"].dt.month
    return data.dropna().copy()


def train_g3() -> dict:
    data = prepare_g3()
    features = [
        "tmax_hier", "tmin_hier", "humidite_hier", "vent_hier",
        "ensoleillement_hier", "pluie_hier", "mois",
    ]
    train = data[data["date"].dt.year <= 2018]
    test = data[data["date"].dt.year >= 2019]
    evaluation = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1).fit(
        train[features], train["temperature_max"]
    )
    prediction = evaluation.predict(test[features])
    residuals = test["temperature_max"].to_numpy() - prediction
    interval = float(np.quantile(np.abs(residuals), 0.9))
    final = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1).fit(
        data[features], data["temperature_max"]
    )
    labels = {
        "tmax_hier": "Température maximale de la veille (°C)",
        "tmin_hier": "Température minimale de la veille (°C)",
        "humidite_hier": "Humidité de la veille (%)",
        "vent_hier": "Vent de la veille",
        "ensoleillement_hier": "Ensoleillement de la veille",
        "pluie_hier": "Pluie de la veille (mm)",
        "mois": "Mois (1 à 12)",
    }
    latest = data.iloc[-1]
    metadata = {
        "id": "g3",
        "title": "Prévision climatique à Lomé",
        "question": "Quelle température maximale le modèle prévoit-il pour le lendemain ?",
        "method": "Forêt aléatoire utilisant les conditions de la veille et le mois.",
        "metrics": {
            "Erreur absolue moyenne 2019–2024 (°C)": finite(mean_absolute_error(test["temperature_max"], prediction)),
            "Référence « comme hier » (°C)": finite(mean_absolute_error(test["temperature_max"], test["tmax_hier"])),
        },
        "fields": numeric_schema(data, features, labels),
        "example": {name: finite(latest[name]) for name in features},
        "interval90": finite(interval),
        "scope": "Prévision à un jour, dans les conditions climatiques représentées entre 1990 et 2024.",
        "warning": "Il s’agit d’un exercice historique, pas d’un bulletin météorologique opérationnel.",
    }
    save("g3", {"model": final, "features": features}, metadata)
    return metadata


def prepare_g4() -> tuple[pd.DataFrame, list[str]]:
    data = pd.read_csv(DATA / "maladie_renale.csv")
    data["diagnostic"] = data["diagnostic"].str.strip()
    for column in ["hypertension", "diabete", "oedemes", "anemie"]:
        data[column] = (data[column].str.strip() == "yes").astype(int)
    data["appetit_faible"] = (data["appetit"].str.strip() == "poor").astype(int)
    data["malade"] = (data["diagnostic"] == "ckd").astype(int)
    data.loc[data["potassium"] > 10, "potassium"] = np.nan
    data.loc[data["sodium"] < 100, "sodium"] = np.nan
    measures = [
        "age", "tension", "densite_urinaire", "albumine_urines", "sucre_urines",
        "glycemie", "uree", "creatinine", "sodium", "potassium", "hemoglobine",
        "hematocrite", "globules_blancs", "globules_rouges",
    ]
    features = measures + ["hypertension", "diabete", "appetit_faible", "oedemes", "anemie"]
    return data, features


def train_g4() -> dict:
    data, features = prepare_g4()
    medians = data[features].median()
    x = data[features].fillna(medians)
    y = data["malade"]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, stratify=y, random_state=42
    )
    evaluation = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1).fit(x_train, y_train)
    prediction = evaluation.predict(x_test)
    final = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1).fit(x, y)
    binary = {"hypertension", "diabete", "appetit_faible", "oedemes", "anemie"}
    labels = {
        "densite_urinaire": "Densité urinaire", "albumine_urines": "Albumine urinaire",
        "sucre_urines": "Sucre urinaire", "globules_blancs": "Globules blancs",
        "globules_rouges": "Globules rouges", "appetit_faible": "Appétit faible",
    }
    fields = numeric_schema(x, features, labels)
    for field in fields:
        if field["name"] in binary:
            field.update({"type": "boolean", "default": int(round(field["default"])), "min": 0, "max": 1})
    metadata = {
        "id": "g4",
        "title": "Exploration du risque rénal",
        "question": "À quel groupe du jeu d’apprentissage ce profil clinique ressemble-t-il le plus ?",
        "method": "Forêt aléatoire sur mesures biologiques et signes cliniques.",
        "metrics": {
            "Exactitude — un découpage test": finite(accuracy_score(y_test, prediction), 4),
            "Sensibilité — un découpage test": finite(recall_score(y_test, prediction), 4),
        },
        "fields": fields,
        "scope": "Démonstration méthodologique sur le jeu Chronic Kidney Disease.",
        "warning": "Ce résultat n’est ni un diagnostic, ni un avis médical. Un score parfait sur ce petit jeu ne constitue pas une validation clinique externe.",
    }
    save("g4", {"model": final, "features": features, "medians": medians.to_dict()}, metadata)
    return metadata


def train_g5() -> dict:
    data = pd.read_csv(DATA / "indicateurs_pays.csv")
    indicators = [
        "esperance_vie", "pib_par_habitant", "acces_eau", "acces_assainissement",
        "acces_electricite", "population_urbaine", "enfants_par_femme", "depenses_sante",
    ]
    clean = data.dropna(subset=indicators).copy()
    clean["log_revenu"] = np.log(clean["pib_par_habitant"])
    features = [
        "log_revenu", "acces_eau", "acces_assainissement", "acces_electricite",
        "population_urbaine", "enfants_par_femme", "depenses_sante",
    ]
    split = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=1)
    train_idx, test_idx = next(split.split(clean, groups=clean["pays"]))
    train, test = clean.iloc[train_idx], clean.iloc[test_idx]
    evaluation = RandomForestRegressor(n_estimators=300, random_state=0, n_jobs=-1).fit(
        train[features], train["esperance_vie"]
    )
    prediction = evaluation.predict(test[features])
    interval = float(np.quantile(np.abs(test["esperance_vie"].to_numpy() - prediction), 0.9))
    final = RandomForestRegressor(n_estimators=300, random_state=0, n_jobs=-1).fit(
        clean[features], clean["esperance_vie"]
    )
    raw_features = [
        "pib_par_habitant", "acces_eau", "acces_assainissement", "acces_electricite",
        "population_urbaine", "enfants_par_femme", "depenses_sante",
    ]
    labels = {
        "pib_par_habitant": "PIB par habitant (dollars PPA)",
        "acces_eau": "Accès à l’eau (%)", "acces_assainissement": "Accès à l’assainissement (%)",
        "acces_electricite": "Accès à l’électricité (%)", "population_urbaine": "Population urbaine (%)",
        "enfants_par_femme": "Nombre d’enfants par femme", "depenses_sante": "Dépenses de santé",
    }
    togo = clean[clean["pays"] == "Togo"].sort_values("annee").iloc[-1]
    metadata = {
        "id": "g5",
        "title": "Indicateurs économiques et espérance de vie",
        "question": "Quelle espérance de vie le modèle associe-t-il à cet ensemble d’indicateurs ?",
        "method": "Forêt aléatoire, évaluée sur des pays tenus à l’écart de l’apprentissage.",
        "metrics": {"Erreur absolue moyenne sur les pays tests (années)": finite(mean_absolute_error(test["esperance_vie"], prediction))},
        "fields": numeric_schema(clean, raw_features, labels),
        "example": {name: finite(togo[name]) for name in raw_features},
        "interval90": finite(interval),
        "scope": "Association prédictive entre indicateurs nationaux et espérance de vie.",
        "warning": "Les variables influentes ne sont pas nécessairement des causes. Le modèle ne simule pas une politique publique.",
    }
    save("g5", {"model": final, "features": features, "raw_features": raw_features}, metadata)
    return metadata


def train_g6() -> dict:
    books = {
        "Equiano": (DATA / "equiano.txt").read_text(encoding="utf-8"),
        "Plaatje": (DATA / "plaatje.txt").read_text(encoding="utf-8"),
        "Conrad": (DATA / "conrad.txt").read_text(encoding="utf-8"),
        "Kingsley": (DATA / "kingsley.txt").read_text(encoding="utf-8"),
    }
    passages, authors = [], []
    for author, text in books.items():
        words = re.findall(r"[A-Za-z']+", text)
        for start in range(0, len(words) - 200, 200):
            passages.append(" ".join(words[start : start + 200]))
            authors.append(author)
    x_train, x_test, y_train, y_test = train_test_split(
        passages, authors, test_size=0.3, stratify=authors, random_state=0
    )
    evaluation = make_pipeline(
        TfidfVectorizer(max_features=3000), LogisticRegression(max_iter=3000)
    ).fit(x_train, y_train)
    prediction = evaluation.predict(x_test)
    final = make_pipeline(
        TfidfVectorizer(max_features=3000), LogisticRegression(max_iter=3000)
    ).fit(passages, authors)
    metadata = {
        "id": "g6",
        "title": "Attribution de textes historiques",
        "question": "De quel ouvrage du corpus ce passage est-il lexicalement le plus proche ?",
        "method": "TF–IDF et régression logistique sur des passages de 200 mots.",
        "metrics": {"Exactitude sur le jeu test": finite(accuracy_score(y_test, prediction), 4)},
        "classes": list(final.named_steps["logisticregression"].classes_),
        "example": passages[0][:1200],
        "scope": "Attribution parmi quatre ouvrages connus du modèle.",
        "warning": "Le résultat ne prouve ni l’identité d’un auteur, ni l’existence d’une « voix » culturelle ou coloniale.",
    }
    save("g6", {"model": final}, metadata)
    return metadata


def main() -> None:
    trainers = [train_g1, train_g2, train_g3, train_g4, train_g5, train_g6]
    catalogue = []
    for trainer in trainers:
        metadata = trainer()
        catalogue.append(metadata)
        print(f"{metadata['id']} — {metadata['title']} : modèle enregistré")
    (MODELS / "catalogue.json").write_text(
        json.dumps(catalogue, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
