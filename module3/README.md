# Formation CNN, API et Docker – Module 3

Supports pratiques du **Module 3** du **Volet IT** de la *Formation intermédiaire et avancée en intelligence artificielle pour les femmes chercheures* (09–13 septembre 2026) : réseaux de neurones, CNN, détection d'objets, déploiement (API REST, Docker) et notions de MLOps.

📑 **Slides du cours** : [dossier Google Drive](https://drive.google.com/drive/folders/1o6uLU0U3MlEFjGYVfoX6h_4q9vKQxJ8A?usp=drive_link)

## Contexte

Cette formation s'inscrit dans le déploiement des formations avancées en numérique porté par le ministère de l'Efficacité du Service public et de la Transformation numérique (MESPTN), à travers l'Agence Togo Digital (ATD) et le Projet de Transformation Digitale de l'Économie Togolaise (ProDigiT). Le ProDigiT est mandaté par le ministère fédéral allemand de la Coopération économique et du Développement (BMZ) et cofinancé par l'Union européenne et le Luxembourg. La formation contribue au programme 8 de la feuille de route *Togo Digital 2025* : la redynamisation d'un écosystème d'innovation et de talents digitaux.

Elle s'adresse aux femmes chercheures togolaises (enseignantes-chercheures, chercheures IT et non-IT), encore largement sous-représentées dans les formations avancées en IA. Elle s'inscrit dans l'approche **« AI Ready »** : une montée en compétences rapide, pour être immédiatement opérationnelle avec l'IA dans son travail quotidien.

Le programme comporte deux parcours, qui partagent un tronc commun (J1 et J5) et se séparent de J2 à J4 :

- **Volet IT**, pour les chercheures qui savent programmer. **Ce dépôt contient les TP de ce volet.**
- **Volet non-IT**, pour les chercheures d'autres disciplines, sans prérequis technique.

La formation est validée par une attestation.

## Objectifs

**Objectif général** : renforcer les compétences des chercheures togolaises en IA, selon leur profil disciplinaire, pour qu'elles puissent concevoir, mettre en œuvre et valoriser des projets de recherche qui intègrent des méthodes et des outils d'IA.

**Tronc commun**
- Comprendre les fondements et les enjeux éthiques de l'IA, y compris les biais de genre
- Intégrer l'IA dans le cycle de la recherche scientifique (veille, analyse, valorisation)
- Développer un réseau professionnel féminin en IA
- Élaborer un plan d'action personnel pour appliquer l'IA dans ses recherches
- Être immédiatement opérationnelle (AI Ready) : IA générative, culture IA, prompt engineering, cas d'usage métiers

**Volet IT**
- Maîtriser les architectures de réseaux de neurones et leur entraînement
- Développer des pipelines NLP et vision par ordinateur appliqués
- Déployer un modèle d'IA léger (API, conteneurisation)

## Contenu des TP

| Lab | Thème | Objectif du Volet IT couvert | Colab |
|-----|-------|------------------------------|-------|
| `00_quick_start` | Prise en main de Colab et de VS Code | — | [![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/atouammi/Formation_CNN_API_Docker_Module4/blob/main/Labs/00_quick_start.ipynb) |
| `01_test_notebook` | Vérification de l'environnement (versions, graphiques) | — | [![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/atouammi/Formation_CNN_API_Docker_Module4/blob/main/Labs/01_test_notebook.ipynb) |
| `02_ml_fundamentals` | Régression linéaire, descente de gradient (par lot et stochastique) | Réseaux de neurones et entraînement (bases) | [![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/atouammi/Formation_CNN_API_Docker_Module4/blob/main/Labs/02_ml_fundamentals.ipynb) |
| `03_neural_nets` | Neurone artificiel, réseaux multicouches, chiffres manuscrits, Keras et PyTorch | Réseaux de neurones et entraînement | [![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/atouammi/Formation_CNN_API_Docker_Module4/blob/main/Labs/03_neural_nets.ipynb) |
| `04_breast_cancer_cnn` | CNN et transfert d'apprentissage (MobileNetV3) pour détecter le cancer du sein sur des échographies (BreastMNIST) ; métriques adaptées aux classes déséquilibrées, choix du seuil, éthique | Réseaux de neurones et vision par ordinateur | [![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/atouammi/Formation_CNN_API_Docker_Module4/blob/main/Labs/04_breast_cancer_cnn.ipynb) |
| `05_yolo_image_detection_example` | Détection d'objets avec YOLOv8 (Ultralytics) | Vision par ordinateur | [![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/atouammi/Formation_CNN_API_Docker_Module4/blob/main/Labs/05_yolo_image_detection_example.ipynb) |
| `api/` *(à venir)* | Exposer un modèle via une API REST, conteneurisation (Docker) | Déploiement d'un modèle léger | — |



## Structure

```
.
├── requirements.txt    # Dépendances Python du cours (local et Colab)
├── Labs/               # Notebooks de travaux pratiques
│   ├── 00_quick_start.ipynb
│   ├── 01_test_notebook.ipynb
│   ├── 02_ml_fundamentals.ipynb
│   ├── 03_neural_nets.ipynb
│   ├── 04_breast_cancer_cnn.ipynb
│   ├── 05_yolo_image_detection_example.ipynb
│   ├── plotting_utils/ # Fonctions de visualisation utilisées par les notebooks
│   └── assets/         # Images d'exemple
└── api/                # Partie API REST / déploiement (à venir)
```

## Exécuter les notebooks

Suivez les notebooks dans l'ordre, en commençant par `00_quick_start.ipynb`. Chaque notebook commence par une cellule **⚙️ Configuration** à exécuter en premier.

### Option A : Google Colab (aucune installation)

Cliquez sur le badge **« Open in Colab »** du notebook dans le tableau ci-dessus, puis exécutez la cellule de configuration. Elle récupère les fichiers du cours depuis GitHub et installe ce qui manque. Colab garde ses propres versions des bibliothèques principales (TensorFlow, PyTorch, NumPy…), ce qui évite une longue réinstallation.

- Pour enregistrer vos modifications : **Fichier → Enregistrer une copie dans Drive**.
- Pour les labs 04 et 05, activez le GPU : **Exécution → Modifier le type d'exécution → GPU (T4)**.

### Option B : en local dans VS Code

Prérequis : **Python 3.11 ou 3.12**, **[VS Code](https://code.visualstudio.com/)** et ses extensions **Python** et **Jupyter**.

```bash
git clone https://github.com/atouammi/Formation_CNN_API_Docker_Module4.git
cd Formation_CNN_API_Docker_Module4
python3 -m venv .venv
source .venv/bin/activate          # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

Ouvrez ensuite le dossier dans VS Code, ouvrez un notebook de `Labs/` et choisissez le noyau `.venv` (**Sélectionner un noyau** en haut à droite).

> **macOS** : si un téléchargement échoue avec `CERTIFICATE_VERIFY_FAILED` (Python installé depuis python.org), exécutez une fois **Applications → Python 3.x → Install Certificates.command**.

## Licence

Voir le fichier [LICENSE](LICENSE).
