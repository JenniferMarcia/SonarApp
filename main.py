from contextlib import asynccontextmanager
import joblib
import os
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from schema import SonarInput


# Dictionnaire global pour stocker les modèles ML chargés en mémoire
ML_MODELS = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        base_dir = os.path.dirname(__file__)
        model_path = os.path.join(base_dir, "models", "best_model_sonar.joblib")
        le_path = os.path.join(base_dir, "models", "label_encoder_sonar.joblib")
       
        if os.path.exists(model_path) and os.path.exists(le_path):
            ML_MODELS["pipeline"] = joblib.load(model_path)
            ML_MODELS["label_encoder"] = joblib.load(le_path)
            print(" Modèles chargés avec succès.")
        else:
            print(" ERREUR : Fichiers modèles introuvables au chemin spécifié.")
    except Exception as e:
        print(f"Erreur critique au chargement : {e}")

    yield
    ML_MODELS.clear()

# On passe le lifespan à l'application
app = FastAPI(lifespan=lifespan)

# Route de prédiction
@app.post("/predict", tags=["Predictions"])
def predict(data: SonarInput):
    """
    Effectue la classification binaire :
    - **M** : Mine (Métal)
    - **R** : Rock (Roche)
    """
    # Vérification de la dimension de l'input (60 colonnes)
    if len(data.features) != 60:
        raise HTTPException(
            status_code=400, 
            detail=f"Le modèle attend 60 caractéristiques, reçu: {len(data.features)}"
        )

    try:
        # Transformation de l'entrée en DataFrame
        feature_names = [f'C{i}' for i in range(1, 61)]
        input_df = pd.DataFrame([data.features], columns=feature_names)

        # Prédiction avec la Pipeline (Scaling + Modèle inclus)
        prediction_idx = ML_MODELS["pipeline"].predict(input_df)[0]
        probabilities = ML_MODELS["pipeline"].predict_proba(input_df)[0]

        # Traduction du code numérique en texte (M ou R)
        classes = ML_MODELS["label_encoder"].classes_
        
        # 3. Traduction du code numérique en texte
        label = ML_MODELS["label_encoder"].inverse_transform([prediction_idx])[0]
        full_label = "Mine (Métal)" if label == 'M' else "Roche"
        
        prob_mapping = {
            str(classes[i]): round(float(probabilities[i]), 4) 
            for i in range(len(classes))
        }

        return {
            "prediction": label,
            "description": full_label,
            "confidence": round(float(np.max(probabilities)), 4),
            "probabilities": prob_mapping
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur interne : {str(e)}")


@app.get("/importance", tags=["Analysis"])
def get_importance():
    """Renvoie l'importance globale des 60 fréquences du modèle RandomForest"""
    if "pipeline" not in ML_MODELS:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    
    try:
        # Extraire le modèle de la pipeline
        model = ML_MODELS["pipeline"].named_steps['clf']
        importances = model.feature_importances_
        
        # Préparer les données pour le front-end
        feature_importance = {
            f"C{i+1}": float(importances[i]) for i in range(len(importances))
        }
        
        return {"feature_importance": feature_importance}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur calcul importance : {str(e)}")

@app.get("/", tags=["System"])
def healthcheck():
    return {"status": "online", "model_loaded": "pipeline" in ML_MODELS}