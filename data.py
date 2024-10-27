import pandas as pd
import numpy as np
from datetime import datetime

#fonction pour générer des données climatiques
def generate_climate_data(start_date, end_date):
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    temperature_avg = np.random.normal(22, 3, len(date_range))  # Réduit la variance pour un climat plus stable
    temperature_min = temperature_avg - np.random.uniform(2, 6, len(date_range))
    temperature_max = temperature_avg + np.random.uniform(2, 6, len(date_range))
    precipitation = np.random.exponential(0.5, len(date_range))  # Faible précipitation pour Sfax
    humidity = np.random.normal(55, 5, len(date_range)).clip(20, 70)  # Moins d'humidité

    climate_data = pd.DataFrame({
        'date': date_range,
        'temperature_avg': temperature_avg,
        'temperature_min': temperature_min,
        'temperature_max': temperature_max,
        'precipitation': precipitation,
        'humidity': humidity
    })
    
    return climate_data

#pour les données sur le sol
def generate_soil_data(num_fields):
    soil_data = pd.DataFrame({
        'field_id': range(1, num_fields + 1),
        'pH': np.random.uniform(7.0, 8.5, num_fields),  # Adapté pour l'olive et les légumes résistants
        'nitrogen': np.random.uniform(30, 70, num_fields),
        'phosphorus': np.random.uniform(20, 60, num_fields),
        'potassium': np.random.uniform(150, 350, num_fields),
        'soil_type': np.random.choice(['sableux', 'limoneux'], num_fields, p=[0.7, 0.3])  # Sfax est majoritairement sableux
    })
    return soil_data

#pour les données sur les cultures
def generate_crop_data(start_date, end_date, num_fields):
    date_range = pd.date_range(start=start_date, end=end_date, freq='M')
    crops = ['oliviers', 'amandiers', 'figuiers', 'tomates', 'poivrons']
    
    crop_data = []
    for field in range(1, num_fields + 1):
        for date in date_range:
            crop = np.random.choice(crops, p=[0.5, 0.2, 0.1, 0.1, 0.1])  # Plus d'oliviers
            yield_value = (
                np.random.normal(1500, 300) if crop == 'oliviers' else
                np.random.normal(1200, 200) if crop == 'amandiers' else
                np.random.normal(800, 150)
            )
            crop_data.append({
                'date': date,
                'field_id': field,
                'crop_type': crop,
                'yield': max(0, yield_value),
                'irrigation': np.random.choice(['goutte-à-goutte', 'aspersion', 'aucune'], p=[0.6, 0.2, 0.2]),
                'fertilizer_used': np.random.choice([True, False], p=[0.7, 0.3])  # Plus d'utilisation d'engrais pour légumes
            })
    
    return pd.DataFrame(crop_data)

#génération des données
start_date = datetime(2020, 1, 1)
end_date = datetime(2023, 12, 31)
num_fields = 50

climate_data = generate_climate_data(start_date, end_date)
soil_data = generate_soil_data(num_fields)
crop_data = generate_crop_data(start_date, end_date, num_fields)

climate_data.to_csv('climate_data_sfax.csv', index=False)
soil_data.to_csv('soil_data_sfax.csv', index=False)
crop_data.to_csv('crop_data_sfax.csv', index=False)

print("Données adaptées à Sfax générées & sauvegardées avec succès.")
