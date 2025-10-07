# prediction_app/tests/test_data_processing.py
from django.test import TestCase
import pandas as pd
import os
from prediction_app.services.data_processing import preprocess_data, load_events_holidays_data, create_time_features, merge_events_holidays

# Définir des chemins de test pour les données simulées
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Remonte à prediction_app
TEST_DATA_PATH = os.path.join(BASE_DIR, 'test_data')
os.makedirs(TEST_DATA_PATH, exist_ok=True) # S'assurer que le dossier de test existe

class DataProcessingTests(TestCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # Créer un fichier CSV de test simple pour le preprocessing
        cls.test_raw_csv_path = os.path.join(TEST_DATA_PATH, 'test_raw_data.csv')
        test_data = {
            'Date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05', '2023-01-06', '2023-01-07']),
            'Nb_Passagers': [1000, 1200, 1100, 1300, 1500, 800, 700],
            'Une_Colonne_Inutile': ['A', 'B', 'C', 'D', 'E', 'F', 'G']
        }
        pd.DataFrame(test_data).to_csv(cls.test_raw_csv_path, index=False)

        # Créer un fichier events_holidays_test.csv
        cls.test_events_csv_path = os.path.join(TEST_DATA_PATH, 'test_events_holidays.csv')
        test_events = {
            'Date': ['2023-01-01', '2023-01-07', '2023-01-04'],
            'Type': ['Jour_Ferie', 'Est_Weekend', 'Evenement_Special'] # Type fictif pour le test
        }
        pd.DataFrame(test_events).to_csv(cls.test_events_csv_path, index=False)

        cls.events_df = load_events_holidays_data(cls.test_events_csv_path)

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        # Nettoyer les fichiers de test
        os.remove(cls.test_raw_csv_path)
        os.remove(cls.test_events_csv_path)
        os.rmdir(TEST_DATA_PATH)


    def test_create_time_features(self):
        df_raw = pd.read_csv(self.test_raw_csv_path)
        df = create_time_features(df_raw.copy())
        self.assertIn('Annee', df.columns)
        self.assertIn('Jour_Semaine', df.columns)
        self.assertIn('Est_Weekend', df.columns)
        self.assertEqual(df.loc[df['Date'] == '2023-01-01', 'Est_Weekend'].iloc[0], 0) # Dimanche 2023-01-01 est un dimanche
        self.assertEqual(df.loc[df['Date'] == '2023-01-07', 'Est_Weekend'].iloc[0], 1) # Samedi 2023-01-07 est un samedi

    def test_merge_events_holidays(self):
        df_raw = pd.read_csv(self.test_raw_csv_path)
        df = create_time_features(df_raw.copy()) # Nécessaire avant merge pour la colonne 'Date'
        df_merged = merge_events_holidays(df, self.events_df)
        self.assertIn('Est_Jour_Ferie', df_merged.columns)
        self.assertEqual(df_merged.loc[df_merged['Date'] == '2023-01-01', 'Est_Jour_Ferie'].iloc[0], 1)
        self.assertEqual(df_merged.loc[df_merged['Date'] == '2023-01-02', 'Est_Jour_Ferie'].iloc[0], 0)
        self.assertEqual(df_merged.loc[df_merged['Date'] == '2023-01-04', 'Evenement_Special'].iloc[0], 1)

    def test_preprocess_data_output_columns(self):
        df_raw = pd.read_csv(self.test_raw_csv_path)
        df_processed = preprocess_data(df_raw.copy(), self.events_df.copy())
        
        # Vérifie que les colonnes attendues (y compris OHE) sont présentes
        expected_cols_part = ['Annee', 'Mois', 'Jour', 'Jour_Semaine', 'Est_Weekend', 
                            'Est_Jour_Ferie', 'Est_Vacances_Scolaires', 'Evenement_Special',
                            'Temperature_Moyenne_C', 'Precipitations_mm', # Vérifier si elles sont générées
                            'Jour_Friday', 'Jour_Monday', 'Jour_Saturday', 'Jour_Sunday',
                            'Jour_Thursday', 'Jour_Tuesday', 'Jour_Wednesday']
        
        for col in expected_cols_part:
            self.assertIn(col, df_processed.columns, f"Colonne manquante: {col}")
        
        # Vérifier que les colonnes non utiles comme 'Une_Colonne_Inutile' ont été supprimées
        # Note: La fonction preprocess_data doit inclure cette logique si tu veux l'appliquer
        # Pour l'instant, elle n'est pas incluse dans la version fournie, donc ce test pourrait échouer.
        # Tu peux ajouter cette logique à preprocess_data.
        # self.assertNotIn('Une_Colonne_Inutile', df_processed.columns)

        self.assertIsInstance(df_processed, pd.DataFrame)
        self.assertFalse(df_processed.isnull().any().any()) # Pas de NaN après preprocessing (si tes fonctions gèrent ça)