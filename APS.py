import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

crop_data = pd.read_csv('/home/haghob/AgriPredictSfax/crop_data_sfax.csv', parse_dates=['date'])
soil_data = pd.read_csv('/home/haghob/AgriPredictSfax/soil_data_sfax.csv')
climate_data = pd.read_csv('/home/haghob/AgriPredictSfax/climate_data_sfax.csv', parse_dates=['date'])



combined_data = crop_data.merge(soil_data, on='field_id').merge(climate_data, on='date')

numeric_features = ['temperature_avg', 'temperature_min', 'temperature_max', 'precipitation', 'humidity', 
                    'pH', 'nitrogen', 'phosphorus', 'potassium']
categorical_features = ['crop_type', 'irrigation', 'soil_type']

combined_data['season'] = pd.cut(combined_data['date'].dt.month, 
                                 bins=[0, 3, 6, 9, 12], 
                                 labels=['Winter', 'Spring', 'Summer', 'Autumn'])
categorical_features.append('season')



X = combined_data[numeric_features + categorical_features]
y = combined_data['yield']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

#création du pipeline complet
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
print(f'Cross-validation R^2 scores: {cv_scores}')
print(f'Mean CV R^2 score: {np.mean(cv_scores)}')

feature_importance = model.named_steps['regressor'].feature_importances_
feature_names = (numeric_features + 
                 model.named_steps['preprocessor']
                     .named_transformers_['cat']
                     .named_steps['onehot']
                     .get_feature_names_out(categorical_features).tolist())

importance_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importance})
importance_df = importance_df.sort_values('importance', ascending=False)

print("\nTop 10 most important features:")
print(importance_df.head(10))