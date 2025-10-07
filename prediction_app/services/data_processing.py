# prediction_app/services/data_processing.py
import pandas as pd
import numpy as np

def load_raw_data(file_path):
    """Charge les données brutes à partir d'un fichier CSV."""
    df = pd.read_csv(file_path)
    return df

def load_events_holidays_data(file_path):
    """Charge les données d'événements et jours fériés."""
    df = pd.read_csv(file_path)
    # Assurez-vous que la colonne 'Date' est de type datetime
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    return df

def create_features(df):
    """
    Crée des caractéristiques à partir de la colonne de date.
    Assure que 'Date' est déjà de type datetime.
    """
    # S'assurer que 'Date' est bien un datetime, car elle pourrait arriver en str du CSV
    if not pd.api.types.is_datetime64_any_dtype(df['Date']):
        df['Date'] = pd.to_datetime(df['Date'])

    df['Annee'] = df['Date'].dt.year
    df['Mois'] = df['Date'].dt.month
    df['Jour'] = df['Date'].dt.day
    df['Jour_Semaine'] = df['Date'].dt.dayofweek # Lundi=0, Dimanche=6
    df['Est_Weekend'] = ((df['Jour_Semaine'] == 5) | (df['Jour_Semaine'] == 6)).astype(int)
    df['Jour_Annee'] = df['Date'].dt.dayofyear # Utile pour les modèles
    df['Semaine_Annee'] = df['Date'].dt.isocalendar().week.astype(int) # Convertir en int

    return df

def merge_events_holidays(df_main, df_events):
    """
    Fusionne les événements/vacances avec le DataFrame principal.
    Crée des colonnes binaires pour les événements/vacances.
    """
    if df_events.empty:
        df_main['Est_Jour_Ferie'] = 0
        df_main['Est_Vacances_Scolaires'] = 0
        df_main['Evenement_Special'] = 0
        return df_main

    # Assure-toi que les colonnes de date sont de type datetime pour la fusion
    df_main['Date_Key'] = df_main['Date'].dt.date
    df_events['Date_Key'] = df_events['Date'].dt.date # Utilise le même nom de colonne pour la fusion

    # Crée des colonnes binaires pour les événements/vacances
    # Cette approche est plus robuste que isin pour de multiples types
    df_main['Est_Jour_Ferie'] = df_main['Date_Key'].isin(df_events[df_events['Type'] == 'Jour_Ferie']['Date_Key']).astype(int)
    df_main['Est_Vacances_Scolaires'] = df_main['Date_Key'].isin(df_events[df_events['Type'] == 'Vacances_Scolaires']['Date_Key']).astype(int)
    df_main['Evenement_Special'] = df_main['Date_Key'].isin(df_events[df_events['Type'] == 'Evenement_Special']['Date_Key']).astype(int)
    # Ajoute ici d'autres types d'événements si tu en as dans ton fichier events_holidays.csv

    # Supprime la colonne temporaire de fusion
    df_main = df_main.drop(columns=['Date_Key'])
    df_events = df_events.drop(columns=['Date_Key'])

    return df_main

def preprocess_data(df_raw, df_events_holidays=None):
    """
    Fonction principale pour charger, nettoyer et créer des caractéristiques
    à partir d'un DataFrame brut et optionnellement d'un DataFrame d'événements.
    """
    df = df_raw.copy()

    # 1. Nettoyage initial et conversion de la date
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce') # Gérer les erreurs de format de date
        df.dropna(subset=['Date'], inplace=True) # Supprimer les lignes où la date est invalide
    else:
        raise ValueError("Le DataFrame doit contenir une colonne 'Date'.")
    
    # 3. Création des caractéristiques temporelles
    df = create_features(df)

    # 4. Intégration des événements et vacances
    if df_events_holidays is not None and not df_events_holidays.empty:
        df = merge_events_holidays(df, df_events_holidays)
    else: # Si pas de fichier d'événements, s'assurer que les colonnes existent et sont à 0
        df['Est_Jour_Ferie'] = 0
        df['Est_Vacances_Scolaires'] = 0
        df['Evenement_Special'] = 0


    # Exemple pour 'Jour_Semaine':
    df = pd.get_dummies(df, columns=['Jour_Semaine'], prefix='Jour')


    return df