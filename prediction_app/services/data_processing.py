# prediction_app/services/data_processing.py
import pandas as pd
import numpy as np
import os

def load_events_holidays_data(file_path):
    """Charge les données d'événements et jours fériés depuis un CSV."""
    if not os.path.exists(file_path):
        print(f"Avertissement: Le fichier d'événements/vacances n'existe pas à {file_path}. Retourne un DataFrame vide.")
        return pd.DataFrame(columns=['Date', 'Type'])
    
    df_events = pd.read_csv(file_path)
    df_events['Date'] = pd.to_datetime(df_events['Date'])
    return df_events

def create_time_features(df):
    """Crée des caractéristiques temporelles à partir de la colonne 'Date'."""
    if 'Date' not in df.columns:
        raise ValueError("Le DataFrame doit contenir une colonne 'Date'.")
        
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Caractéristiques temporelles
    df['Annee'] = df['Date'].dt.year
    df['Mois'] = df['Date'].dt.month
    df['Jour'] = df['Date'].dt.day
    df['Jour_Semaine'] = df['Date'].dt.dayofweek # Lundi=0, Dimanche=6
    df['Est_Weekend'] = ((df['Jour_Semaine'] == 5) | (df['Jour_Semaine'] == 6)).astype(int)
    
    # Si tu as une granularité horaire, tu pourrais ajouter :
    # df['Heure'] = df['Date'].dt.hour
    # df['Minute'] = df['Date'].dt.minute
    
    return df

def merge_events_holidays(df_main, df_events_holidays):
    """Fusionne les événements et jours fériés au DataFrame principal."""
    # S'assurer que les colonnes 'Date' sont de type date pour la fusion
    df_main_dates = df_main['Date'].dt.date
    df_events_dates = df_events_holidays['Date'].dt.date

    # Initialiser les colonnes d'événements à 0
    df_main['Est_Jour_Ferie'] = 0
    df_main['Est_Vacances_Scolaires'] = 0
    df_main['Evenement_Special'] = 0

    # Mettre à jour les colonnes d'événements basées sur le fichier de référence
    for event_type in df_events_holidays['Type'].unique():
        event_dates = df_events_holidays[df_events_holidays['Type'] == event_type]['Date'].dt.date
        if event_type == 'Jour_Ferie':
            df_main.loc[df_main_dates.isin(event_dates), 'Est_Jour_Ferie'] = 1
        elif event_type == 'Vacances_Scolaires':
            df_main.loc[df_main_dates.isin(event_dates), 'Est_Vacances_Scolaires'] = 1
        elif event_type == 'Evenement_Special':
            df_main.loc[df_main_dates.isin(event_dates), 'Evenement_Special'] = 1
        # Ajoute d'autres types d'événements si tu en as dans events_holidays.csv

    return df_main

def create_weather_features(df):
    """
    Crée des caractéristiques météorologiques (simulées) si elles ne sont pas déjà présentes.
    Dans un vrai scénario, ces données viendraient d'une API météo ou d'un dataset.
    Pour la prédiction, il faut s'assurer que ces colonnes existeront pour les dates futures.
    """
    if 'Temperature_Moyenne_C' not in df.columns:
        # Simuler la température (basé sur le mois)
        df['Temperature_Moyenne_C'] = (15 + 10 * np.sin((df['Mois'] - 3) * (2 * np.pi / 12)) +
                                       np.random.normal(0, 3, len(df)))
        df['Temperature_Moyenne_C'] = df['Temperature_Moyenne_C'].round(1)

    if 'Precipitations_mm' not in df.columns:
        # Simuler les précipitations
        df['Precipitations_mm'] = np.where(
            df['Mois'].isin([10, 11, 12, 1, 2, 3]), # Mois pluvieux
            np.random.normal(3, 5, len(df)),
            np.random.normal(0.5, 1.5, len(df))
        )
        df['Precipitations_mm'] = np.maximum(0, df['Precipitations_mm']).round(1)
    
    return df

def preprocess_data(df_raw, df_events_holidays=None):
    """
    Fonction principale de prétraitement des données.
    Prend un DataFrame brut (issu de l'upload) et un DataFrame d'événements/vacances.
    Retourne un DataFrame prêt pour la prédiction.
    """
    df = df_raw.copy()

    # Nettoyage initial : Renommer la colonne Date si elle a un nom différent
    # ou s'assurer qu'elle existe. Pour le CSV uploadé, on suppose 'Date'.
    if 'Date' not in df.columns:
        raise ValueError("Le fichier CSV uploadé doit contenir une colonne 'Date'.")
    
    # Gérer les valeurs manquantes (exemple: si des passagers sont manquants dans les données historiques)
    # Pour un fichier de prédiction, 'Nb_Passagers' ne sera pas présent ou sera la cible.
    if 'Nb_Passagers' in df.columns:
        df.dropna(subset=['Nb_Passagers'], inplace=True) # ou imputer si pertinent
        # Renommer la colonne cible pour éviter la confusion avec les features
        df.rename(columns={'Nb_Passagers': 'Passagers_Reels'}, inplace=True)
    else:
        df['Passagers_Reels'] = np.nan # Ajoute une colonne vide pour la cohérence

    df = create_time_features(df)
    
    if df_events_holidays is not None and not df_events_holidays.empty:
        df = merge_events_holidays(df, df_events_holidays)
    
    # Crée les features météo si elles ne sont pas déjà dans le CSV uploadé
    df = create_weather_features(df)

    # Convertir les jours de semaine en One-Hot Encoding si tes modèles les attendent ainsi
    # C'est une étape CRUCIALE pour la cohérence entre entraînement et prédiction
    # Assure-toi que les colonnes 'Jour_Lundi', 'Jour_Mardi', etc. sont créées
    # avec toutes les 7 colonnes, même si un jour n'est pas présent dans le DF
    df = pd.get_dummies(df, columns=['Jour_Semaine'], prefix='Jour')

    # Gérer les colonnes après One-Hot Encoding pour s'assurer d'avoir toutes les 7 colonnes
    # C'est un point de défaillance courant si le DF de prédiction n'a pas tous les jours de semaine.
    all_days = ['Jour_Monday', 'Jour_Tuesday', 'Jour_Wednesday', 'Jour_Thursday', 'Jour_Friday', 'Jour_Saturday', 'Jour_Sunday']
    for day in all_days:
        if day not in df.columns:
            df[day] = 0


    # Supprimer la colonne 'Date' si elle n'est plus nécessaire après l'extraction des caractéristiques temporelles
    # ou si elle sera utilisée comme index dans les graphiques
    # Gardons-la pour le moment pour les graphiques dans Django
    # df = df.drop(columns=['Date']) 

    return df

# --- Bloc de test (peut être supprimé ou commenté après validation) ---
if __name__ == '__main__':
    # Chemin vers ton fichier de données brutes simulées (passengers_casatramway_raw.csv)
    # Assure-toi que ce fichier a été généré via ton script initial
    RAW_DATA_TEST_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'raw', 'passengers_casatramway_raw.csv')
    EVENTS_HOLIDAYS_TEST_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'raw', 'events_holidays.csv')

    print(f"Chargement des données brutes depuis: {RAW_DATA_TEST_PATH}")
    print(f"Chargement des événements/vacances depuis: {EVENTS_HOLIDAYS_TEST_PATH}")

    # Test de la fonction load_events_holidays_data
    events_df = load_events_holidays_data(EVENTS_HOLIDAYS_TEST_PATH)
    print("\nDataFrame d'événements/vacances (head):")
    print(events_df.head())

    try:
        df_raw_test = pd.read_csv(RAW_DATA_TEST_PATH)
        print("\nDataFrame brut (head) avant preprocessing:")
        print(df_raw_test.head())
        print("\nDataFrame brut (info) avant preprocessing:")
        print(df_raw_test.info())

        df_processed_test = preprocess_data(df_raw_test.copy(), events_df.copy())

        print("\nDataFrame traité (head) après preprocessing:")
        print(df_processed_test.head())
        print("\nDataFrame traité (info) après preprocessing:")
        print(df_processed_test.info())
        print("\nColonnes du DataFrame traité :")
        print(df_processed_test.columns.tolist())

        # Vérifier que les colonnes de jours de semaine sont bien là
        print("\nExemple de vérification des colonnes One-Hot Encoded pour les jours de semaine :")
        print(df_processed_test[[col for col in df_processed_test.columns if col.startswith('Jour_')]].head())

    except FileNotFoundError:
        print(f"Erreur: Le fichier {RAW_DATA_TEST_PATH} n'a pas été trouvé. Veuillez le générer d'abord.")
    except Exception as e:
        print(f"Une erreur est survenue lors du test du preprocessing: {e}")