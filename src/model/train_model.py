import os
from model import AgriPredictSfaxModel
from joblib import dump, load

if __name__ == "__main__":
    aps_model = AgriPredictSfaxModel()
    MODEL_PATH = '/home/haghob/AgriPredictSfax/model.pkl'  
    
    try:
        if os.path.exists(MODEL_PATH):
            print(f"Chargement du modèle à partir de {MODEL_PATH}...")
            aps_model.model = load(MODEL_PATH)  
        else:
            print("Chargement des données...")
            combined_data = aps_model.load_data()
            
            print("Préparation des caractéristiques...")
            X, y = aps_model.prepare_features(combined_data)
            
            print("Création du pipeline...")
            aps_model.create_pipeline()
            
            print("Entraînement du modèle...")
            aps_model.model.fit(X, y)  
            
            print("Sauvegarde du modèle...")
            dump(aps_model.model, MODEL_PATH)
            print(f"Modèle sauvegardé sous {MODEL_PATH}")

    except Exception as e:
        print(f"Une erreur est survenue: {str(e)}")
