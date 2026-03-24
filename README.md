---
title: Sonar App
emoji: 🐳
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

Sonar-project est unr application d'**Intelligence Artificielle** pour la classification binaire de signaux sonar :
- **M** : Mine
- **R** : Roche

Application complète avec :
- **Backend** : FastAPI pour les prédictions ML
- **Frontend** : Streamlit pour l'interface utilisateur

**Meilleur modèle** : RandomForest (Pipeline avec StandardScaler) - **F1-score ~83%** (CV 5-fold).

##  Structure du projet
```
Sonar/
├── app.py              # Interface Streamlit (UI)
├── main.py             # API FastAPI (Backend ML)
├── schema.py           # Modèles Pydantic
├── requirements.txt    # Dépendances
├── Solar_notebook.ipynb # Notebook complet (EDA + Entraînement)
├── dataset/
│   └── sonar.all-data.csv
├── models/
│   ├── best_model_sonar.joblib
│   ├── label_encoder_sonar.joblib
└── README.md
```

##  Démarrage rapide

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Lancer l'API (Backend)
```bash
uvicorn main:app --reload --port 8000
```
- Docs API : http://127.0.0.1:8000/docs
- Healthcheck : http://127.0.0.1:8000/

### 3. Lancer l'interface (Frontend)
```bash
streamlit run app.py
```
- [Ouvrir dans le navigateur](http://localhost:8501)

## Utilisation
1. **Générer des données simulées** (60 fréquences sonar) dans la sidebar Streamlit.
2. **Cliquer "Lancer l'analyse via l'API"**.
3. **Résultats** : Prédiction, confiance, graphiques (signal + probabilités).

**Exemple d'entrée API** :
```json
POST http://127.0.0.1:8000/predict
{
  "features": [0.02, 0.0453, ... ]  // 60 valeurs [0-1]
}
```

**Réponse** :
```json
{
  "prediction": "M",
  "description": "Mine (Métal)",
  "confidence": 0.92,
  "probabilities": {"Mine (M)": 0.92, "Rock (R)": 0.08}
}
```

##  Performances du modèle
| Modèle             | F1-score (CV) | Précision Test | Rappel Test |
|--------------------|---------------|----------------|-------------|
| RandomForest (best)| **83%**       | 84%            | 83%         |
| LogisticRegression | 80%           | 81%            | 80%         |
| DecisionTree       | 78%           | 79%            | 78%         |

##  Notebook d'entraînement
`Solar_notebook.ipynb` contient :
- EDA (histogrammes, boxplots, heatmap corrélation)
- Préprocessing (LabelEncoder, StandardScaler)
- GridSearchCV sur 3 modèles
- Sauvegarde des artefacts


