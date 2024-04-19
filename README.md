# Projet Deep Learning

## Diogo Almeida

## Caractéristiques

- **ACCEUIL**: Une page d'introduction qui présente l'application.
- **GET DATA**: Permet à l'utilisateur de récupérer les dernières données AQI pour toutes les capitales listées.
- **MAP**: Affiche une carte interactive montrant les capitales avec des marqueurs colorés en fonction de l'AQI.
- **GRAPHS**: Présente des visualisations sous forme de graphiques pour une analyse approfondie.

### Créer un Environnement Virtuel

Naviguez dans le répertoire du projet via votre terminal. Et faite,

Pour les utilisateurs Unix/MacOS :

```sh
python3 -m venv venv
source venv/bin/activate
```

Pour les utilisateurs Windows :

```sh
python -m venv venv
.\venv\Scripts\activate
```

## Installer les Dépendances
Installez toutes les dépendances requises avec le fichier requirements.txt :

```sh
pip install -r requirements.txt
```
## Lancer l'Application

- **1**: Clonez ce dépôt sur votre machine locale.
- **2**: Naviguez dans le répertoire du projet via votre terminal.
- **3**: Activez votre environnement virtuel.
- **4**: Exécutez l'application en utilisant la commande suivante :

```sh
streamlit run main.py
```