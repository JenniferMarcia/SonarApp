from contextlib import asynccontextmanager
import joblib
import os
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from schema import SonarInput
import shap
import matplotlib.pyplot as plt
import io
import base64
from fastapi.responses import FileResponse
import logging


from config import (
    MODEL_PATH,
    LABEL_ENCODER_PATH,
    MONITORING_FILE,
    CLASSES,
    N_FEATURES,
    FEATURE_NAMES,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Dictionnaire global pour stocker les modèles ML chargés en mémoire
ML_MODELS = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gère le cycle de vie de l'application FastAPI.
    """
    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(LABEL_ENCODER_PATH):
            ML_MODELS["pipeline"] = joblib.load(MODEL_PATH)
            ML_MODELS["label_encoder"] = joblib.load(LABEL_ENCODER_PATH)      
            
            # Initialiser l'explainer SHAP
            model_core = ML_MODELS["pipeline"].named_steps['clf']
            ML_MODELS["explainer"] = shap.TreeExplainer(model_core)
            
            logger.info("Modèles chargés avec succès.")
        else:
            logger.error("ERREUR : Fichiers modèles introuvables au chemin spécifié.")
         
    except Exception as e:
        logger.error(f"Erreur critique au chargement : {e}")

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
    if len(data.features) != N_FEATURES:
        raise HTTPException(
            status_code=400, 
            detail=f"Le modèle attend 60 caractéristiques, reçu: {len(data.features)}"
        )

    try:
        # Transformation de l'entrée en DataFrame
        input_df = pd.DataFrame([data.features], columns=FEATURE_NAMES)

        # Prédiction avec la Pipeline (Scaling + Modèle inclus)
        prediction_idx = ML_MODELS["pipeline"].predict(input_df)[0]
        probabilities = ML_MODELS["pipeline"].predict_proba(input_df)[0]

        # Traduction du code numérique en texte (M ou R)
        classes = ML_MODELS["label_encoder"].classes_
        
        # 3. Traduction du code numérique en texte
        label = ML_MODELS["label_encoder"].inverse_transform([prediction_idx])[0]
        full_label = CLASSES[label]
        
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


@app.post("/explain", tags=["Analysis"])
def explain_prediction(data: SonarInput):
    """Génère une explication SHAP (SHapley Additive exPlanations) pour une prédiction.
    """
    if "pipeline" not in ML_MODELS or "explainer" not in ML_MODELS:
        raise HTTPException(status_code=503, detail="Modèle ou explainer non chargé")
    
    if len(data.features) != N_FEATURES:
        raise HTTPException(status_code=400, detail=f"Attendu {N_FEATURES} features")
    
    try:
        input_df = pd.DataFrame([data.features], columns=FEATURE_NAMES)
        
        # Scaler
        scaler = ML_MODELS["pipeline"].named_steps['scaler']
        X_scaled = scaler.transform(input_df)
        X_scaled_1d = X_scaled[0]
        
        # SHAP - juste calculer
        explainer = ML_MODELS["explainer"]
        shap_values = explainer.shap_values(X_scaled)
               
        # Format binaire : extraire la classe Mine (index 1)
        if isinstance(shap_values, list):
            # Format list [array_classe_0, array_classe_1]
            shap_vals_1d = shap_values[1][0]
            base_value = explainer.expected_value[1]
        elif len(shap_values.shape) == 3:
            # Format array (échantillons, caractéristiques, classes)
            shap_vals_1d = shap_values[0, :, 1]
            base_value = explainer.expected_value[1]
        else:
            # Cas array simple
            shap_vals_1d = shap_values[0]
            base_value = explainer.expected_value

        plt.figure(figsize=(10, 4))
        shap.force_plot(
            base_value, 
            shap_vals_1d, 
            X_scaled_1d,
            feature_names=FEATURE_NAMES, 
            matplotlib=True, 
            show=False
            )
        
        # En base64
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches='tight', dpi=100)
        plt.close()
        img_str = base64.b64encode(buf.getvalue()).decode()
        
        return {"shap_plot": img_str}
    
    except Exception as e:
        logger.info(f"[EXPLAIN ERROR] {str(e)}")
        import traceback
        logger.info(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/", tags=["System"])
def healthcheck():
    """
    Vérifie l'état de santé de l'API et des composants associés.
    """
    return {
        "status": "online", 
        "model_loaded": "pipeline" in ML_MODELS,
        "monitoring_exists": os.path.exists(MONITORING_FILE)}

    
@app.get("/monitoring", tags=["Monitoring"])
def get_monitoring():
    """
    Télécharge le rapport de monitoring pré-généré
    Le fichier se trouve dans monitoring/monitoring_train_test.html
    """
    try:
        # Vérifier si le fichier existe
        if not os.path.exists(MONITORING_FILE):
            logger.warning(f"Fichier monitoring non trouvé : {MONITORING_FILE}")
            raise HTTPException(
                status_code=404, 
                detail="Rapport de monitoring non trouvé. Veuillez d'abord générer le rapport."
            )  
        # Retourner le fichier
        return FileResponse(
            path=MONITORING_FILE,
            filename="monitoring_sonar.html",
            media_type='text/html'
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur téléchargement monitoring : {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur : {str(e)}")