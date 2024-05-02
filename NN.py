import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

data = pd.read_csv('aggregated_data.csv')

data['Date'] = pd.to_datetime(data['Date'])
data['Jour'] = data['Date'].dt.day
data['Mois'] = data['Date'].dt.month
data['Année'] = data['Date'].dt.year
data['Heure'] = data['Date'].dt.hour
data['Minutes'] = data['Date'].dt.minute

encoder = OneHotEncoder()
city_encoded = encoder.fit_transform(data[['City']])
city_df = pd.DataFrame(city_encoded.toarray(), columns=encoder.get_feature_names_out(['City']))

feature_cols = ['Longitude', 'Latitude', 'Jour', 'Mois', 'Année', 'Heure', 'Minutes'] + list(city_df.columns)

X = pd.concat([data[feature_cols[:-len(city_df.columns)]], city_df], axis=1)
y = data['Value']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def build_model(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_dim=input_dim),
        Dense(32, activation='relu', input_dim = 100),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

model = build_model(X_train.shape[1])
model.fit(X_train, y_train, epochs=50, batch_size=10, validation_data=(X_test, y_test))


def adjust_features(features, expected_features_count):
    # Vérifie si des caractéristiques sont manquantes
    missing_features = expected_features_count - features.shape[1]
    if missing_features > 0:
        # Ajouter les caractéristiques manquantes comme des zéros
        missing_matrix = np.zeros((features.shape[0], missing_features))
        features = np.hstack([features, missing_matrix])
    elif missing_features < 0:
        # Supprimer les caractéristiques excédentaires (si nécessaire)
        features = features[:, :expected_features_count]
    return features

# Utilisation de la fonction ajustée dans predict_aqi
def predict_aqi(city, date, encoder, model):
    date_parsed = pd.to_datetime(date)
    date_features = [date_parsed.year, date_parsed.month, date_parsed.day, date_parsed.hour, date_parsed.minute]
    
    city_array = encoder.transform([[city]]).toarray()[0]
    
    features = np.hstack([date_features, city_array])
    features = np.array(features).reshape(1, -1)
    
    # Ajuster les caractéristiques pour correspondre aux attentes du modèle
    features = adjust_features(features, 2587)
    
    predicted_value = model.predict(features)
    return predicted_value[0][0]