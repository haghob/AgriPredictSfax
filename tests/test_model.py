import joblib
import numpy as np
import pandas as pd

MODEL_PATH = '/home/haghob/AgriPredictSfax/model.joblib'

def test_model():
    model = joblib.load(MODEL_PATH)
    
    sample_data = pd.DataFrame([{
    'temperature_avg': 25.5,
    'temperature_min': 20.0,
    'temperature_max': 31.0,
    'precipitation': 10.5,
    'humidity': 65,
    'pH': 6.8,
    'nitrogen': 40,
    'phosphorus': 35,
    'potassium': 45,
    'crop_type': "blé",
    'irrigation': "drip",
    'soil_type': "loam",
    'drought_index': 0.5,  
    'season': 'summer',     
    'matiere_organique': 1.5,  
    'profondeur_sol': 60,   
    'fertilizer_used': True, 
    'calcium': 1500,        
    'day_of_year': 180,     
    'temp_humidity_interaction': 15,  
    'npk_ratio': 1.2        
}])


    prediction = model.predict(sample_data)
    print(f'Prédiction : {prediction[0]}')

if __name__ == '__main__':
    test_model()
