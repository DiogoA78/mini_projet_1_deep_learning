import requests
import pandas as pd
import streamlit as st
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from datetime import datetime
import matplotlib.pyplot as plt
from NN import predict_aqi


def collect_aqi_data(capital_list, api_url, api_key):
    aqi_data = []
    for city_name in capital_list:
        response = requests.get(api_url.format(city_name, api_key))
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "ok":
                aqi = data.get('data', {}).get('aqi')
                geo = data.get('data', {}).get('city', {}).get('geo', [None, None])
                latitude, longitude = geo
                
                dominentpol = data.get('data', {}).get('dominentpol')
                #iaqi = data.get('data', {}).get('iaqi', {})
                time_last_updated = data.get('data', {}).get('time', {}).get('s')
                
                city_data = {
                    'Capital': city_name,
                    'AQI': aqi,
                    'Latitude': latitude,
                    'Longitude': longitude,
                    'Dominant Pollutant': dominentpol,
                    'Time of Last Update': time_last_updated,
                    #'Details': iaqi
                }
                aqi_data.append(city_data)
            else:
                aqi_data.append({'Capital': city_name, 'AQI': 'Non répertorié', 'Latitude': None, 'Longitude': None})
        else:
            print(f"Erreur lors de la requête pour {city_name} : Statut {response.status_code}")
            aqi_data.append({'Capital': city_name, 'AQI': 'Erreur de requête', 'Latitude': None, 'Longitude': None})
    return pd.DataFrame(aqi_data)

Capital_df = pd.read_csv("country-list.csv")

# Configuration de l'API
api_url = "http://api.waqi.info/feed/{}/?token={}"
api_key = "7d53b1063e3f113e01ff8189283f1578da695000"

menu = ["ACCUEIL", "GET DATA", "MAP", "GRAPHS", "PREDICTION"]
choice = st.sidebar.selectbox("Menu", menu)

# Page ACCUEIL
if choice == "ACCUEIL":
    st.title("Bienvenue sur l'application de Visualisation de l'Indice de Qualité de l'Air (AQI)")

# Page GET DATA
elif choice == "GET DATA":
    st.title("Collecte des données AQI")
    
    if st.button('Collecter les données AQI'):
        if 'data_loaded' not in st.session_state or not st.session_state['data_loaded']:
            progress_bar = st.progress(0)
            total = len(Capital_df['capital'])
            for i, city_name in enumerate(Capital_df['capital']):
                progress_bar.progress((i + 1) / total)
                aqi_df = collect_aqi_data([city_name], api_url, api_key)
                if i == 0:
                    all_aqi_df = aqi_df
                else:
                    all_aqi_df = pd.concat([all_aqi_df, aqi_df], ignore_index=True)
            aqi_df = collect_aqi_data(Capital_df['capital'], api_url, api_key)
            aqi_df['AQI'] = pd.to_numeric(aqi_df['AQI'], errors='coerce')
            aqi_df_valid = aqi_df.dropna(subset=['AQI'])
            aqi_df_valid.reset_index(drop=True, inplace=True)
            st.session_state['aqi_data'] = aqi_df_valid
            st.session_state['data_loaded'] = True
            st.success("Les données AQI ont été collectées et chargées avec succès!")
        else:
            st.info("Les données AQI sont déjà chargées. Recliquez si vous souhaitez les recharger.")

    if st.button('Recharger les données AQI'):
        progress_bar = st.progress(0)
        total = len(Capital_df['capital'])
        for i, city_name in enumerate(Capital_df['capital']):
            progress_bar.progress((i + 1) / total)
            aqi_df = collect_aqi_data([city_name], api_url, api_key)
            if i == 0:
                all_aqi_df = aqi_df
            else:
                all_aqi_df = pd.concat([all_aqi_df, aqi_df], ignore_index=True)
        aqi_df = collect_aqi_data(Capital_df['capital'], api_url, api_key)
        aqi_df['AQI'] = pd.to_numeric(aqi_df['AQI'], errors='coerce')
        aqi_df_valid = aqi_df.dropna(subset=['AQI'])
        aqi_df_valid.reset_index(drop=True, inplace=True)
        st.session_state['aqi_data'] = aqi_df_valid
        st.session_state['data_loaded'] = True
        st.success("Les données AQI ont été rechargées avec succès!")

# Page MAP
elif choice == "MAP":
    st.title("Carte de l'Indice de Qualité de l'Air (AQI)")
    if 'aqi_data' in st.session_state:
        aqi_df_valid = st.session_state['aqi_data']
        m = folium.Map(location=[48.8566, 2.3522], zoom_start=6)
        marker_cluster = MarkerCluster().add_to(m)
        for idx, row in aqi_df_valid.iterrows():
            folium.CircleMarker(
                location=(row['Latitude'], row['Longitude']),
                radius=10,
                color='green' if row['AQI'] <= 50 else 'orange' if row['AQI'] <= 100 else 'red',
                fill=True,
                fill_color='green' if row['AQI'] <= 50 else 'orange' if row['AQI'] <= 100 else 'red',
                popup=folium.Popup(f"{row['Capital']} - AQI: {row['AQI']}", parse_html=True)
            ).add_to(marker_cluster)
        st_folium(m, width=725, height=500)

# Page GRAPHS
elif choice == "GRAPHS":
    st.title("Visualisations des Indices de Qualité de l'Air")
    if 'aqi_data' in st.session_state:
        aqi_df_valid = st.session_state['aqi_data']

        st.write("Villes et leur Indice de Qualité de l'Air (AQI)")
        st.table(aqi_df_valid)

        # Histogramme des AQI
        st.subheader("Histogramme de la Distribution des AQI")
        fig, ax = plt.subplots()
        aqi_df_valid['AQI'].plot(kind='hist', ax=ax, bins=30, color='blue', edgecolor='black')
        ax.set_xlabel("AQI")
        ax.set_ylabel("Fréquence")
        ax.set_title("Distribution des AQI")
        st.pyplot(fig)

        # Graphique en barres des AQI par capitale
        st.subheader("AQI par Capitale")
        sorted_aqi = aqi_df_valid.sort_values('AQI', ascending=True)
        fig, ax = plt.subplots(figsize=(8, 10))
        ax.barh(sorted_aqi['Capital'], sorted_aqi['AQI'], color='green')
        ax.set_xlabel("AQI")
        ax.set_title("AQI par Capitale")
        ax.tick_params(axis='y', labelsize=8)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

        # Graphique en nuage de points des AQI en fonction de la latitude et longitude
        st.subheader("AQI en fonction de la position géographique")
        fig, ax = plt.subplots()
        scatter = ax.scatter(aqi_df_valid['Longitude'], aqi_df_valid['Latitude'], c=aqi_df_valid['AQI'], cmap='viridis')
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title("AQI en fonction de la Longitude et Latitude")
        colorbar = fig.colorbar(scatter)
        colorbar.set_label('AQI')
        st.pyplot(fig)


elif choice == "PREDICTION":
    st.title("Prédiction de la Qualité de l'Air")
    city = st.selectbox('Sélectionnez une ville', options=pd.read_csv("AQI.csv")['City'].unique())
    date = st.date_input("Sélectionnez une date", min_value=datetime(2024, 5, 1))
    if st.button('Prédire'):
        prediction = predict_aqi(city, date)
        st.write(f"La valeur prédite de l'AQI pour {city} le {date} est {prediction}")