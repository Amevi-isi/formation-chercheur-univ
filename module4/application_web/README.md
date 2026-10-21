# Laboratoire IA — J4 ProDigiT

Application web unique pour expérimenter les six modèles des notebooks de la journée J4.

## Principes

- Les notebooks sources ne sont pas modifiés.
- Les modèles sont entraînés par `train_models.py` à partir des mêmes données et des mêmes choix méthodologiques.
- Les modèles destinés à l’interface sont ensuite réentraînés sur toutes les données exploitables.
- Les métriques affichées restent calculées sur les découpages de validation des notebooks.
- Le TP1 reste agrégé et le TP4 est explicitement non diagnostique.

## Installation locale

### Méthode la plus simple sous Windows

Double-cliquer une première fois sur `INSTALLER_ET_LANCER.bat`. Pour les utilisations suivantes, double-cliquer sur `LANCER.bat`.

### Installation manuelle

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe train_models.py
.\.venv\Scripts\python.exe run_app.py
```

Ouvrir ensuite `http://127.0.0.1:8000`.

Les autres ordinateurs connectés au même réseau peuvent utiliser l’adresse réseau affichée dans la fenêtre de lancement. Le pare-feu Windows peut demander une autorisation lors du premier démarrage.

## Déploiement avec Docker

Après avoir entraîné les modèles :

```powershell
docker compose up --build -d
```

L’application est alors disponible sur le port `8000` de la machine. Sur un réseau local, les participantes peuvent utiliser l’adresse IP de cette machine.

## Données attendues pour l’entraînement

Le dossier `data/` doit contenir :

- `afrobarometre_togo_2024.csv`
- `haricots.csv`
- `meteo_lome.csv`
- `maladie_renale.csv`
- `indicateurs_pays.csv`
- `equiano.txt`, `plaatje.txt`, `conrad.txt`, `kingsley.txt`

## Limite de portée

Cette application est un prototype pédagogique déployable. Elle n’est ni un dispositif médical, ni un système de décision publique, ni une preuve de causalité.
