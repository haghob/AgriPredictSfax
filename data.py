import pandas as pd
import numpy as np
from datetime import datetime

# Fonction pour générer des données climatiques
def generate_climate_data(start_date, end_date):
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Génération de données climatiques réalistes pour Sfax
    temperature_avg = np.random.normal(22, 5, len(date_range))
    temperature_min = temperature_avg - np.random.uniform(3, 7, len(date_range))
    temperature_max = temperature_avg + np.random.uniform(3, 7, len(date_range))
    precipitation = np.random.exponential(1, len(date_range))
    humidity = np.random.normal(60, 10, len(date_range)).clip(0, 100)

    # Création du DataFrame avec les données générées
    climate_data = pd.DataFrame({
        'date': date_range,
        'temperature_avg': temperature_avg,
        'temperature_min': temperature_min,
        'temperature_max': temperature_max,
        'precipitation': precipitation,
        'humidity': humidity
    })
    
    return climate_data

# Fonction pour générer des données sur le sol
def generate_soil_data(num_fields):
    soil_data = pd.DataFrame({
        'field_id': range(1, num_fields + 1),
        'pH': np.random.uniform(6.0, 8.0, num_fields),
        'nitrogen': np.random.uniform(20, 80, num_fields),
        'phosphorus': np.random.uniform(10, 50, num_fields),
        'potassium': np.random.uniform(100, 300, num_fields),
        'soil_type': np.random.choice(['sableux', 'argileux', 'limoneux'], num_fields)
    })
    return soil_data

# Fonction pour générer des données sur les cultures
def generate_crop_data(start_date, end_date, num_fields):
    date_range = pd.date_range(start=start_date, end=end_date, freq='M')
    crops = ['blé', 'orge', 'oliviers', 'amandiers']
    
    crop_data = []
    for field in range(1, num_fields + 1):
        for date in date_range:
            crop = np.random.choice(crops)
            yield_value = np.random.normal(3000, 500) if crop in ['blé', 'orge'] else np.random.normal(1500, 300)
            crop_data.append({
                'date': date,
                'field_id': field,
                'crop_type': crop,
                'yield': max(0, yield_value),
                'irrigation': np.random.choice(['goutte-à-goutte', 'aspersion', 'aucune']),
                'fertilizer_used': np.random.choice([True, False])
            })
    
    return pd.DataFrame(crop_data)

# Génération des données
start_date = datetime(2020, 1, 1)
end_date = datetime(2023, 12, 31)
num_fields = 50

climate_data = generate_climate_data(start_date, end_date)
soil_data = generate_soil_data(num_fields)
crop_data = generate_crop_data(start_date, end_date, num_fields)

climate_data.to_csv('climate_data_sfax.csv', index=False)
soil_data.to_csv('soil_data_sfax.csv', index=False)
crop_data.to_csv('crop_data_sfax.csv', index=False)

print("Données générées et sauvegardées avec succès.")
