import pandas as pd
import numpy as np
from datetime import datetime

def generate_climate_data(start_date, end_date):
    """Génère des données climatiques typiques de Sfax."""
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    #variation saisonnière des températures à Sfax
    days = np.arange(len(date_range))
    annual_cycle = np.sin(2 * np.pi * days / 365)
    
    #températures typiques de Sfax (été chaud, hiver doux)
    temperature_avg = 22 + 8 * annual_cycle + np.random.normal(0, 2, len(date_range))
    temperature_min = temperature_avg - np.random.uniform(3, 7, len(date_range))
    temperature_max = temperature_avg + np.random.uniform(5, 10, len(date_range))
    
    #précipitations saisonnières (plus de pluie en hiver)
    winter_mask = (date_range.month >= 11) | (date_range.month <= 2)
    precipitation = np.zeros(len(date_range))
    precipitation[winter_mask] = np.random.exponential(2, sum(winter_mask))
    precipitation[~winter_mask] = np.random.exponential(0.3, sum(~winter_mask))
    
    #humidité relative (plus élevée en hiver)
    base_humidity = 55 + 10 * annual_cycle
    humidity = base_humidity + np.random.normal(0, 5, len(date_range)).clip(30, 80)

    climate_data = pd.DataFrame({
        'date': date_range,
        'temperature_avg': temperature_avg,
        'temperature_min': temperature_min,
        'temperature_max': temperature_max,
        'precipitation': precipitation,
        'humidity': humidity,
        'wind_speed': np.random.normal(15, 5, len(date_range)).clip(0, 30)  #vent typique de Sfax
    })
    
    return climate_data

def generate_soil_data(num_fields):
    """Génère des données sur les sols typiques de Sfax."""
    #distribution plus réaliste des types de sol de Sfax
    soil_types = np.random.choice(
        ['sableux', 'sablo-limoneux', 'limoneux', 'argilo-calcaire'],
        num_fields,
        p=[0.4, 0.3, 0.2, 0.1]
    )
    
    #caractéristiques du sol adaptées à chaque type
    ph_values = []
    nitrogen_values = []
    phosphorus_values = []
    potassium_values = []
    calcium_values = []
    
    for soil_type in soil_types:
        if soil_type == 'sableux':
            ph_values.append(np.random.uniform(7.2, 8.0))
            nitrogen_values.append(np.random.uniform(30, 50))
            phosphorus_values.append(np.random.uniform(20, 40))
            potassium_values.append(np.random.uniform(150, 250))
            calcium_values.append(np.random.uniform(2000, 3000))
        elif soil_type == 'sablo-limoneux':
            ph_values.append(np.random.uniform(7.5, 8.2))
            nitrogen_values.append(np.random.uniform(40, 60))
            phosphorus_values.append(np.random.uniform(30, 50))
            potassium_values.append(np.random.uniform(200, 300))
            calcium_values.append(np.random.uniform(2500, 3500))
        else:  # limoneux ou argilo-calcaire
            ph_values.append(np.random.uniform(7.8, 8.5))
            nitrogen_values.append(np.random.uniform(50, 70))
            phosphorus_values.append(np.random.uniform(40, 60))
            potassium_values.append(np.random.uniform(250, 350))
            calcium_values.append(np.random.uniform(3000, 4000))
    
    soil_data = pd.DataFrame({
        'field_id': range(1, num_fields + 1),
        'soil_type': soil_types,
        'pH': ph_values,
        'nitrogen': nitrogen_values,
        'phosphorus': phosphorus_values,
        'potassium': potassium_values,
        'calcium': calcium_values,
        'profondeur_sol': np.random.uniform(30, 120, num_fields),  # en cm
        'matiere_organique': np.random.uniform(0.5, 2.0, num_fields)  # pourcentage
    })
    
    return soil_data

def generate_crop_data(start_date, end_date, num_fields):
    """Génère des données sur les cultures typiques de Sfax."""
    date_range = pd.date_range(start=start_date, end=end_date, freq='M')
    
    #cultures typiques de Sfax avec leurs probabilités
    crops = {
        'oliviers': 0.6,      # Culture dominante
        'amandiers': 0.15,    # Deuxième culture importante
        'pistachiers': 0.05,  # Culture émergente
        'figuiers': 0.05,     # Culture traditionnelle
        'grenadiers': 0.05,   # Culture adaptée au climat
        'tomates': 0.05,      # Culture maraîchère
        'poivrons': 0.05      # Culture maraîchère
    }
    
    #rendements typiques par culture (kg/ha)
    yields = {
        'oliviers': (1200, 300),      # (moyenne, écart-type)
        'amandiers': (800, 200),
        'pistachiers': (600, 150),
        'figuiers': (1000, 250),
        'grenadiers': (1500, 300),
        'tomates': (25000, 5000),
        'poivrons': (20000, 4000)
    }
    
    crop_data = []
    for field in range(1, num_fields + 1):
        #chaque champ garde la même culture pendant toute la période
        field_crop = np.random.choice(
            list(crops.keys()),
            p=list(crops.values())
        )
        
        for date in date_range:
            #ajustement saisonnier du rendement
            season_factor = 1.0
            if field_crop in ['tomates', 'poivrons']:
                #meilleur rendement au printemps/été
                season_factor = 1.2 if date.month in [4, 5, 6, 7, 8, 9] else 0.8
            
            base_yield, std = yields[field_crop]
            yield_value = max(0, np.random.normal(
                base_yield * season_factor,
                std
            ))
            
            #système d'irrigation adapté à chaque culture
            if field_crop in ['oliviers', 'amandiers', 'pistachiers']:
                irrigation_probs = [0.7, 0.2, 0.1]  # goutte-à-goutte, aspersion, aucune
            else:
                irrigation_probs = [0.8, 0.2, 0.0]  # cultures maraîchères nécessitent plus d'irrigation
            
            crop_data.append({
                'date': date,
                'field_id': field,
                'crop_type': field_crop,
                'yield': yield_value,
                'irrigation': np.random.choice(
                    ['goutte-à-goutte', 'aspersion', 'aucune'],
                    p=irrigation_probs
                ),
                'fertilizer_used': np.random.choice([True, False], p=[0.8, 0.2]),
                'age_plantation': np.random.randint(1, 50) if field_crop in ['oliviers', 'amandiers'] else 1
            })
    
    return pd.DataFrame(crop_data)

def add_seasonal_effects(climate_data, crop_data):
    """Ajoute des effets saisonniers sur les rendements."""
    merged_data = crop_data.merge(climate_data, on='date')
    
    #ajustement des rendements en fonction des conditions climatiques
    for idx, row in merged_data.iterrows():
        # Stress thermique
        if row['temperature_max'] > 40:
            merged_data.loc[idx, 'yield'] *= 0.8
        
        #effet précipitations
        if row['precipitation'] < 0.1:
            merged_data.loc[idx, 'yield'] *= 0.9
        
        #effet humidité
        if row['humidity'] < 40:
            merged_data.loc[idx, 'yield'] *= 0.95
    
    return merged_data[crop_data.columns]

#génération des données
if __name__ == "__main__":
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2023, 12, 31)
    num_fields = 50

    print("Génération des données climatiques...")
    climate_data = generate_climate_data(start_date, end_date)
    
    print("Génération des données sur le sol...")
    soil_data = generate_soil_data(num_fields)
    
    print("Génération des données sur les cultures...")
    crop_data = generate_crop_data(start_date, end_date, num_fields)
    
    print("Application des effets saisonniers...")
    crop_data = add_seasonal_effects(climate_data, crop_data)

    climate_data.to_csv('climate_data_sfax.csv', index=False)
    soil_data.to_csv('soil_data_sfax.csv', index=False)
    crop_data.to_csv('crop_data_sfax.csv', index=False)

    print("Données générées et sauvegardées avec succès!")
    
    print("\nStatistiques des données générées:")
    print("\nDistribution des types de cultures:")
    print(crop_data['crop_type'].value_counts(normalize=True))
    
    print("\nMoyenne des rendements par culture:")
    print(crop_data.groupby('crop_type')['yield'].mean())
    
    print("\nDistribution des types de sol:")
    print(soil_data['soil_type'].value_counts(normalize=True))