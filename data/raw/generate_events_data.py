# generate_events_data.py
import pandas as pd
import os

# Définis les chemins et noms de fichiers
output_path = './data/raw/'
output_filename = 'events_holidays.csv'

# Jours Fériés 2022-2024
jours_feries_2022_2024 = [
    pd.to_datetime('2022-01-01'), pd.to_datetime('2022-01-11'), pd.to_datetime('2022-05-01'),
    pd.to_datetime('2022-07-30'), pd.to_datetime('2022-08-14'), pd.to_datetime('2022-08-20'),
    pd.to_datetime('2022-08-21'), pd.to_datetime('2022-11-06'), pd.to_datetime('2022-11-18'),
    pd.to_datetime('2023-01-01'), pd.to_datetime('2023-01-11'), pd.to_datetime('2023-05-01'),
    pd.to_datetime('2023-07-30'), pd.to_datetime('2023-08-14'), pd.to_datetime('2023-08-20'),
    pd.to_datetime('2023-08-21'), pd.to_datetime('2023-11-06'), pd.to_datetime('2023-11-18'),
    pd.to_datetime('2024-01-01'), pd.to_datetime('2024-01-11')
]

# Périodes de Vacances Scolaires
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

# Événements spéciaux simulés
evenements_speciaux = [
    pd.to_datetime('2022-03-25'), pd.to_datetime('2022-05-15'), pd.to_datetime('2022-09-05'),
    pd.to_datetime('2023-03-25'), pd.to_datetime('2023-05-15'), pd.to_datetime('2023-09-05')
]

# Créer un DataFrame pour les événements et jours fériés
events_list = []

for date in jours_feries_2022_2024:
    events_list.append({'Date': date, 'Type': 'Jour_Ferie'})

for start_date, end_date in vacances_scolaires_periods:
    current_date = start_date
    while current_date <= end_date:
        events_list.append({'Date': current_date, 'Type': 'Vacances_Scolaires'})
        current_date += pd.Timedelta(days=1)

for date in evenements_speciaux:
    events_list.append({'Date': date, 'Type': 'Evenement_Special'})

df_events = pd.DataFrame(events_list)
df_events['Date'] = df_events['Date'].dt.strftime('%Y-%m-%d') # Format string pour la sauvegarde CSV

# Enregistrer le fichier
os.makedirs(output_path, exist_ok=True)
df_events.to_csv(os.path.join(output_path, output_filename), index=False)

print(f"Fichier d'événements et jours fériés créé et sauvegardé sous : {os.path.join(output_path, output_filename)}")
print("\nPremières lignes du jeu de données d'événements :")
print(df_events.head())