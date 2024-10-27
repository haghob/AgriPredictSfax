## CONTEXTE DU PROJET : 

Le projet APS (AgriPredictSfax) est un projet MLOps en agriculture, basé à Sfax, pourrait être axé sur la prédiction des rendements agricoles ou la surveillance des conditions climatiques pour optimiser la production. 


## Projet MLOps : Prédiction des Rendements Agricoles à Sfax

### Objectif :

Utilisation des données météorologiques locales, des données sur la qualité du sol, et des informations sur les cultures pour prédire les rendements agricoles à chaque saison. Cela aiderait à planifier les semis, à optimiser les ressources, et à maximiser la production.

### Étapes du projet :

#### ÉTAPE 1 : Collecte des données agricoles locales


### Données climatiques (climate_data_sfax.csv): 

Récupère les données historiques de température, précipitations, et humidité de Sfax via des API météorologiques comme OpenWeatherMap.

### Données sur le sol (soil_data_sfax) : 
Obtiens des données sur la qualité du sol (pH, nutriments, etc.) et les types de cultures plantées dans la région.

### Données sur les cultures (crop_data_sfax.cxv) : 
Enregistre les informations sur les types de cultures plantées et leur rendement au cours des dernières saisons.

**Remarque : J'ai généré les données nécessaires pour mon projet (fichier data.py) en raison de l'insuffisance de données disponibles**

#### ÉTAPE 2 : Préparation des données 

On va intégrer et nettoyer ces données pour en faire un jeu de données exploitable pour l'entraînement d'un modèle prédictif.


### ÉTAPE 3 : Entraînement d’un modèle de prédiction de rendement agricole

On peut utiliser un modèle de régression linéaire ou un Random Forest pour prédire le rendement basé sur les données climatiques et les caractéristiques du sol.



```bash
AgriPredictSfax/
├── data/
│   ├── raw/                   #données brutes
│   │   ├── crop_data_sfax.csv
│   │   ├── soil_data_sfax.csv
│   │   └── climate_data_sfax.csv
│   └── processed/            #données prétraitées
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── data_generation.py     
│   ├── model/
│   │   ├── __init__.py
│   │   └── model.py              #classe AgriPredictSfaxModel
│   └── api/
│       ├── __init__.py
│       └── app.py                #API Flask
├── notebooks/
│   └── APS.ipynb               #notebook de développement
├── tests/
│   ├── __init__.py
│   ├── test_data.py
│   ├── test_model.py
│   └── test_api.py
├── config/
│   └── config.yaml
├── requirements.txt
└── README.md
```


#### ÉTAPE 4 : Mise en place du pipeline MLOps

Pour assurer une automatisation et une gestion continue du projet, voici les outils qu'on peut utiliser :

DVC pour versionner les données et suivre les performances des modèles.

Docker pour encapsuler le modèle de prédiction et ses dépendances.

GitHub Actions pour automatiser l’entraînement du modèle à chaque nouvelle saison.

Exemple de pipeline avec DVC :
```bash

dvc init
dvc add climate_data_sfax.csv soil_quality.csv crop_yield.csv
git add climate_data_sfax.csv.dvc soil_quality.csv.dvc crop_yield.csv.dvc .gitignore
git commit -m "Ajout des données pour la prédiction de rendement agricole"
```

Dockerfile pour encapsuler le projet :
dockerfile

```bash
FROM python:3.8-slim
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . /app
WORKDIR /app
CMD ["python", "train_model.py"]
```

#### ÉTAPE 5 : Surveiller les prédictions et ajuster le modèle

Après avoir mis en place le modèle, on peut utiliser des outils comme Prometheus et Grafana pour surveiller la précision des prédictions en temps réel et réentraîner le modèle régulièrement.

RECAP : 

Impact réel sur le projet agricole

Prédiction des rendements : On peut anticiper les récoltes et ajuster les ressources (eau, engrais) pour maximiser le rendement.

Optimisation des plantations : Les données nous permettront de choisir les périodes idéales pour les semis en fonction des conditions climatiques prévues.

Gestion des risques : En cas de conditions météorologiques défavorables, on peut réagir plus rapidement et prendre des mesures pour protéger les cultures.

Ce projet améliorerait considérablement la gestion de la production agricole tout en exploitant les compétences en data et MLOps.



***************************************************************




Données Climatiques

OOpenWeatherMap API : API gratuite pour obtenir des données météorologiques historiques et actuelles.

Météo Tunisie : Consulte le site de l'Office National de la Météorologie (ONM) pour des données historiques sur le climat en Tunisie.

Données sur le Sol
FAO (Organisation des Nations Unies pour l'alimentation et l'agriculture) : Propose des données sur la qualité du sol, les nutriments, etc.

FAO
Institut National de la Statistique (INS) : Pour des données sur l'agriculture et les terres cultivées en Tunisie.

INS Tunisie
Données sur les Cultures et Rendements
Ministère de l'Agriculture en Tunisie : Récupère des données sur les types de cultures, les rendements et les pratiques agricoles.

Données Open Data : Consulte des plateformes comme data.gov.tn pour des ensembles de données ouvertes, y compris celles liées à l'agriculture.

Autres Sources
Kaggle : Pour des datasets sur les rendements agricoles, les prix des produits, etc. (bien que ce soit souvent général, certains ensembles peuvent contenir des informations sur la région).
Kaggle Datasets
