"""
Configuration centrale pour l'application SonarApp.
Tous les chemins et paramètres sont centralisés ici.
"""

import os
from pathlib import Path

# Chemin racine du projet
BASE_DIR = Path(__file__).parent

# ==================== CHEMINS FICHIERS ====================
# Modèles
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "best_model_sonar.joblib"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder_sonar.joblib"

# Données
DATASET_DIR = BASE_DIR / "dataset"
REFERENCE_DATA = DATASET_DIR / "sonar.all-data.csv"
HISTORY_DATA = DATASET_DIR / "current_signals.csv"

# Monitoring
MONITORING_DIR = BASE_DIR / "monitoring"
MONITORING_FILE = MONITORING_DIR / "monitoring_train_test.html"

# ==================== PARAMÈTRES API ====================
API_HOST = "localhost"
API_PORT = 8000
API_URL = f"http://{API_HOST}:{API_PORT}"

# ==================== PARAMÈTRES MODÈLE ====================
N_FEATURES = 60
FEATURE_NAMES = [f'C{i}' for i in range(1, N_FEATURES + 1)]
CLASSES = {
    'M': 'Mine (Métal)',
    'R': 'Roche'
}

# ==================== CRÉATION DES DOSSIERS ====================
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(MONITORING_DIR, exist_ok=True)