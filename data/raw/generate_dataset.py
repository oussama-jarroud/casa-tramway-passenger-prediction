import pandas as pd
import numpy as np
from datetime import timedelta
import os # Ajout pour la création du dossier

# --- Configuration des Paramètres ---
start_date = pd.to_datetime('2022-01-01')
end_date = pd.to_datetime('2024-01-31') # Environ 2 ans de données + 1 mois
output_filename = 'passengers_casatramway_raw.csv'
output_path = './data/raw/' # Assure-toi que ce dossier existe

# --- Jours Fériés et Périodes de Vacances au Maroc (Exemples Simplifiés) ---
# Ceci est une simplification. Dans un vrai projet, il faudrait une liste plus exhaustive et dynamique.
jours_feries_2022_2024 = [
    pd.to_datetime('2022-01-01'), pd.to_datetime('2022-01-11'), pd.to_datetime('2022-05-01'),
    pd.to_datetime('2022-07-30'), pd.to_datetime('2022-08-14'), pd.to_datetime('2022-08-20'),
    pd.to_datetime('2022-08-21'), pd.to_datetime('2022-11-06'), pd.to_datetime('2022-11-18'),
    pd.to_datetime('2023-01-01'), pd.to_datetime('2023-01-11'), pd.to_datetime('2023-05-01'),
    pd.to_datetime('2023-07-30'), pd.to_datetime('2023-08-14'), pd.to_datetime('2023-08-20'),
    pd.to_datetime('2023-08-21'), pd.to_datetime('2023-11-06'), pd.to_datetime('2023-11-18'),
    pd.to_datetime('2024-01-01'), pd.to_datetime('2024-01-11')
]

vacances_scolaires_periods = [
    (pd.to_datetime('2022-01-20'), pd.to_datetime('2022-01-30')), # Janvier
    (pd.to_datetime('2022-03-20'), pd.to_datetime('2022-04-03')), # Printemps
    (pd.to_datetime('2022-06-25'), pd.to_datetime('2022-09-01')), # Été
    (pd.to_datetime('2022-10-20'), pd.to_datetime('2022-10-30')), # Octobre
    (pd.to_datetime('2022-12-20'), pd.to_datetime('2023-01-05')), # Hiver
    (pd.to_datetime('2023-01-20'), pd.to_datetime('2023-01-30')), # Janvier
    (pd.to_datetime('2023-03-20'), pd.to_datetime('2023-04-03')), # Printemps
    (pd.to_datetime('2023-06-25'), pd.to_datetime('2023-09-01')), # Été
    (pd.to_datetime('2023-10-20'), pd.to_datetime('2023-10-30')), # Octobre
    (pd.to_datetime('2023-12-20'), pd.to_datetime('2024-01-05')) # Hiver
]

# Quelques événements spéciaux simulés (ex: matchs de foot importants, concerts)
evenements_speciaux = [
    pd.to_datetime('2022-03-25'), pd.to_datetime('2022-05-15'), pd.to_datetime('2022-09-05'),
    pd.to_datetime('2023-03-25'), pd.to_datetime('2023-05-15'), pd.to_datetime('2023-09-05')
]

# --- Création de la Séquence de Dates ---
dates = pd.date_range(start=start_date, end=end_date, freq='D')
df = pd.DataFrame(dates, columns=['Date'])

# --- Génération des Caractéristiques Temporelles ---
df['Jour_Semaine'] = df['Date'].dt.day_name()
df['Mois'] = df['Date'].dt.month
df['Annee'] = df['Date'].dt.year
df['Est_Week_End'] = df['Jour_Semaine'].isin(['Saturday', 'Sunday']).astype(int)

# --- Génération des Caractéristiques "Jours Fériés" et "Vacances Scolaires" ---
df['Est_Jour_Ferie'] = df['Date'].isin(jours_feries_2022_2024).astype(int)

df['Est_Vacances_Scolaires'] = 0
for start, end in vacances_scolaires_periods:
    df.loc[(df['Date'] >= start) & (df['Date'] <= end), 'Est_Vacances_Scolaires'] = 1

# --- Génération des Caractéristiques "Événements Spéciaux" ---
df['Evenement_Special'] = df['Date'].isin(evenements_speciaux).astype(int)

# --- Génération des Caractéristiques Météorologiques (Simulées) ---
# Température (variation saisonnière + bruit)
df['Temperature_Moyenne_C'] = (15 + 10 * np.sin((df['Mois'] - 3) * (2 * np.pi / 12)) +
                               np.random.normal(0, 3, len(df)))
df['Temperature_Moyenne_C'] = df['Temperature_Moyenne_C'].round(1)

# Précipitations (plus élevées en hiver)
df['Precipitations_mm'] = np.where(
    df['Mois'].isin([10, 11, 12, 1, 2, 3]), # Mois pluvieux
    np.random.normal(3, 5, len(df)),
    np.random.normal(0.5, 1.5, len(df))
)
df['Precipitations_mm'] = np.maximum(0, df['Precipitations_mm']).round(1) # Pas de précipitations négatives

# --- Génération du Nombre de Passagers (Simulé) ---
# Base de passagers par jour de semaine (plus élevés en semaine, plus bas le week-end)
base_passagers_jour = {
    'Monday': 60000, 'Tuesday': 62000, 'Wednesday': 61000, 'Thursday': 63000,
    'Friday': 58000, 'Saturday': 35000, 'Sunday': 28000
}
df['Nb_Passagers_Base'] = df['Jour_Semaine'].map(base_passagers_jour)

# Ajout de bruit aléatoire
df['Nb_Passagers'] = df['Nb_Passagers_Base'] + np.random.normal(0, 5000, len(df))

# Impact des Jours Fériés et Vacances (réduction significative)
num_jours_feries = (df['Est_Jour_Ferie'] == 1).sum()
if num_jours_feries > 0:
    df.loc[df['Est_Jour_Ferie'] == 1, 'Nb_Passagers'] = \
        df.loc[df['Est_Jour_Ferie'] == 1, 'Nb_Passagers'] * np.random.uniform(0.4, 0.6, num_jours_feries)

num_vacances_scolaires = (df['Est_Vacances_Scolaires'] == 1).sum()
if num_vacances_scolaires > 0:
    df.loc[df['Est_Vacances_Scolaires'] == 1, 'Nb_Passagers'] = \
        df.loc[df['Est_Vacances_Scolaires'] == 1, 'Nb_Passagers'] * np.random.uniform(0.7, 0.9, num_vacances_scolaires)

# Impact des Événements Spéciaux (augmentation)
num_evenements_speciaux = (df['Evenement_Special'] == 1).sum()
if num_evenements_speciaux > 0:
    df.loc[df['Evenement_Special'] == 1, 'Nb_Passagers'] = \
        df.loc[df['Evenement_Special'] == 1, 'Nb_Passagers'] * np.random.uniform(1.2, 1.5, num_evenements_speciaux)

# Impact des Températures (léger boost par temps chaud, légère baisse par temps froid extrême)
df['Nb_Passagers'] = np.where(
    df['Temperature_Moyenne_C'] > 25,
    df['Nb_Passagers'] * np.random.uniform(1.05, 1.15, len(df)), # Correction ici
    df['Nb_Passagers']
)
df['Nb_Passagers'] = np.where(
    df['Temperature_Moyenne_C'] < 5,
    df['Nb_Passagers'] * np.random.uniform(0.9, 0.95, len(df)), # Correction ici
    df['Nb_Passagers']
)

# Impact des Précipitations (légère baisse par temps de pluie)
df['Nb_Passagers'] = np.where(
    df['Precipitations_mm'] > 5,
    df['Nb_Passagers'] * np.random.uniform(0.9, 0.98, len(df)), # Correction ici
    df['Nb_Passagers']
)

# Assurer que le nombre de passagers est un entier positif
df['Nb_Passagers'] = np.maximum(0, df['Nb_Passagers'].round(0)).astype(int)

# Supprimer la colonne temporaire
df = df.drop(columns=['Nb_Passagers_Base'])

# --- Sauvegarde du Dataset ---
# Créer le dossier data/raw s'il n'existe pas
os.makedirs(output_path, exist_ok=True)

df.to_csv(os.path.join(output_path, output_filename), index=False)

print(f"Jeu de données simulé créé et sauvegardé sous : {os.path.join(output_path, output_filename)}")
print("\nPremières lignes du jeu de données :")
print(df.head())
print("\nInformations générales :")
print(df.info())