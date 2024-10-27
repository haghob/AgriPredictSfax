from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import logging
from datetime import datetime
from typing import Dict, List, Union
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app) 

MODEL_PATH = '/home/haghob/AgriPredictSfax/model.pkl'
REQUIRED_FEATURES = [
    'temperature_avg', 'temperature_min', 'temperature_max', 'precipitation', 
    'humidity', 'pH', 'nitrogen', 'phosphorus', 'potassium',
    'crop_type', 'irrigation', 'soil_type'
]

def load_model():
    """Charge le modèle ML depuis le fichier pickle."""
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        logger.error(f"Erreur lors du chargement du modèle: {str(e)}")
        raise

def validate_input(data: Dict) -> bool:
    """
    Valide les données d'entrée.
    
    Args:
        data (Dict): Données à valider
        
    Returns:
        bool: True si les données sont valides
    """
    if not isinstance(data, dict):
        return False
    
    for feature in REQUIRED_FEATURES:
        if feature not in data:
            return False
    
    return True

def preprocess_data(data: Dict) -> pd.DataFrame:
    """
    Prétraite les données pour la prédiction.
    
    Args:
        data (Dict): Données à prétraiter
        
    Returns:
        pd.DataFrame: DataFrame prétraité
    """
    df = pd.DataFrame([data])
    
    if 'date' in data:
        df['date'] = pd.to_datetime(df['date'])
        df['season'] = pd.cut(df['date'].dt.month, 
                            bins=[0, 3, 6, 9, 12],
                            labels=['Winter', 'Spring', 'Summer', 'Autumn'])
    
    return df

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint de vérification de l'état de l'API."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint de prédiction.
    
    Exemple de requête:
    {
        "temperature_avg": 25.5,
        "temperature_min": 20.0,
        "temperature_max": 31.0,
        "precipitation": 10.5,
        "humidity": 65,
        "pH": 6.8,
        "nitrogen": 40,
        "phosphorus": 35,
        "potassium": 45,
        "crop_type": "blé",
        "irrigation": "drip",
        "soil_type": "loam"
    }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Aucune donnée reçue'}), 400
        
        if not validate_input(data):
            return jsonify({'error': 'Données invalides ou incomplètes'}), 400
        
        df = preprocess_data(data)
        
        model = load_model()
        prediction = model.predict(df)
        
        logger.info(f"Prédiction réussie - Input: {data}, Output: {prediction[0]}")
        
        response = {
            'prediction': float(prediction[0]),
            'timestamp': datetime.now().isoformat(),
            'input_data': data
        }
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """
    Endpoint pour les prédictions par lots.
    
    Exemple de requête:
    [
        {
            "temperature_avg": 25.5,
            ...
        },
        {
            "temperature_avg": 26.0,
            ...
        }
    ]
    """
    try:
        data = request.get_json()
        if not isinstance(data, list):
            return jsonify({'error': 'Les données doivent être une liste'}), 400
        
        for entry in data:
            if not validate_input(entry):
                return jsonify({'error': 'Données invalides dans le lot'}), 400
        
        df = pd.DataFrame(data)
        
        model = load_model()
        predictions = model.predict(df)
        
        response = {
            'predictions': predictions.tolist(),
            'timestamp': datetime.now().isoformat(),
            'count': len(predictions)
        }
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction par lots: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint non trouvé'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Erreur interne du serveur'}), 500

if __name__ == '__main__':
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Modèle non trouvé à l'emplacement: {MODEL_PATH}")
        raise FileNotFoundError(f"Le fichier modèle {MODEL_PATH} n'existe pas")
    
    try:
        model = load_model()
        logger.info("Modèle chargé avec succès")
    except Exception as e:
        logger.error(f"Échec du chargement du modèle: {str(e)}")
        raise
    
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
    
    app.run(host='0.0.0.0', port=port, debug=debug)