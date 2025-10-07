# prediction_app/views.py
from django.shortcuts import render
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage
import pandas as pd
import joblib
import os
import json

# Importe tes fonctions de preprocessing depuis le dossier 'services'
from .services.data_processing import preprocess_data, load_events_holidays_data

# Définis les chemins vers tes modèles et tes données brutes/événements
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Remonte à la racine du projet Django
MODEL_DIR = os.path.join(BASE_DIR, 'saved_models')
RAW_DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'passengers_casatramway_raw.csv') # Données historiques pour référence
EVENTS_HOLIDAYS_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'events_holidays.csv')
MEDIA_ROOT_DIR = os.path.join(BASE_DIR, 'media') # Chemin pour le stockage des uploads

# Charger les données d'événements et jours fériés une seule fois au démarrage de l'app
try:
    EVENTS_HOLIDAYS_DF = load_events_holidays_data(EVENTS_HOLIDAYS_PATH)
except Exception as e:
    EVENTS_HOLIDAYS_DF = pd.DataFrame() # DataFrame vide si erreur
    print(f"Attention: Erreur lors du chargement de {EVENTS_HOLIDAYS_PATH} : {e}. Les événements/vacances ne seront pas inclus.")

# Charger les modèles une seule fois au démarrage
MODELS = {}
try:
    MODELS['XGBoost'] = joblib.load(os.path.join(MODEL_DIR, 'xgboost_model.pkl'))
    MODELS['Random Forest'] = joblib.load(os.path.join(MODEL_DIR, 'random_forest_model.pkl'))
    MODELS['Linear Regression'] = joblib.load(os.path.join(MODEL_DIR, 'linear_regression_model.pkl'))
    print("Modèles ML chargés avec succès !")
except FileNotFoundError as e:
    print(f"Erreur lors du chargement d'un modèle : {e}. Assurez-vous que les fichiers .pkl existent dans {MODEL_DIR}")
except Exception as e:
    print(f"Une erreur inattendue est survenue lors du chargement des modèles : {e}")

# Définir les noms de colonnes attendues par tes modèles après preprocessing
# C'est TRÈS IMPORTANT que cette liste corresponde exactement aux colonnes du DF d'entraînement
# Tu devrais générer cette liste après avoir entraîné tes modèles et l'enregistrer.
# Pour l'instant, c'est un exemple basé sur le preprocessing de ton code.
EXPECTED_FEATURES = [
    'Annee', 'Mois', 'Jour', 'Jour_Semaine', 'Est_Weekend',
    'Est_Jour_Ferie', 'Est_Vacances_Scolaires', 'Evenement_Special',
    'Temperature_Moyenne_C', 'Precipitations_mm',
    'Jour_Friday', 'Jour_Monday', 'Jour_Saturday', 'Jour_Sunday',
    'Jour_Thursday', 'Jour_Tuesday', 'Jour_Wednesday'
    # Ajoute ici toutes les autres caractéristiques que tu as créées et utilisées pour l'entraînement
    # Par exemple, les lags si tu les as inclus: 'Lag_Passagers_1_jour', 'Rolling_Mean_7_days'
]


def predict_view(request):
    context = {}
    if request.method == 'POST' and request.FILES.get('csv_file'):
        uploaded_file = request.FILES['csv_file']
        fs = FileSystemStorage(location=MEDIA_ROOT_DIR)
        filename = fs.save(uploaded_file.name, uploaded_file)
        file_path = fs.path(filename)

        try:
            # 1. Charger le CSV uploadé
            df_uploaded = pd.read_csv(file_path)

            # 2. Appliquer le preprocessing
            # On utilise la fonction preprocess_data qui s'attend à un DataFrame
            df_processed = preprocess_data(df_uploaded.copy(), EVENTS_HOLIDAYS_DF.copy())
            
            # S'assurer que 'Date' est en datetime pour l'indexation future si besoin
            df_processed['Date'] = pd.to_datetime(df_processed['Date'])

            # 3. Sélectionner le modèle et faire la prédiction
            model_name = request.POST.get('model_choice', 'XGBoost')
            model = MODELS.get(model_name)

            if model is None:
                context['error'] = f"Le modèle '{model_name}' n'est pas disponible ou n'a pas pu être chargé."
                return render(request, 'prediction_app/prediction_form.html', context)
            
            # S'assurer que le DataFrame passé au modèle contient uniquement les features attendues
            # et qu'elles sont dans le bon ordre.
            # Crée un DataFrame de features pour la prédiction
            df_features_for_prediction = df_processed[EXPECTED_FEATURES]

            predictions = model.predict(df_features_for_prediction)
            df_processed['Predictions'] = predictions.tolist() # Convertir numpy array en list pour JSON

            # 4. Préparer les données pour le template (JSON)
            df_processed['Date'] = df_processed['Date'].dt.strftime('%Y-%m-%d') # Convertir dates en string pour JSON

            chart_data = df_processed[['Date', 'Passagers_Reels', 'Predictions']].to_dict(orient='records')
            
            context = {
                'model_used': model_name,
                'predictions_df': df_processed.to_html(classes='table table-striped', index=False),
                'chart_data_json': json.dumps(chart_data)
            }

        except ValueError as e:
            context['error'] = f"Erreur de données: {e}. Vérifiez le format de votre CSV (doit contenir une colonne 'Date' et optionnellement 'Nb_Passagers')."
        except Exception as e:
            context['error'] = f"Une erreur est survenue lors du traitement : {e}"
        finally:
            # Supprimer le fichier uploadé après traitement
            if os.path.exists(file_path):
                fs.delete(filename)

    # Charger les données historiques globales pour un affichage permanent si désiré
    # ou pour peupler le graphique initial
    try:
        df_history_raw = pd.read_csv(RAW_DATA_PATH)
        # Utilise aussi le preprocessing pour les données historiques pour la cohérence
        df_history_processed = preprocess_data(df_history_raw.copy(), EVENTS_HOLIDAYS_DF.copy())
        
        # Ajuste le nom de la colonne cible pour l'affichage
        if 'Nb_Passagers' in df_history_processed.columns:
             df_history_processed.rename(columns={'Nb_Passagers': 'Passagers_Reels'}, inplace=True)
        
        df_history_processed['Date'] = pd.to_datetime(df_history_processed['Date']) # S'assurer du type datetime
        df_history_processed['Date'] = df_history_processed['Date'].dt.strftime('%Y-%m-%d') # Convertir en string pour JSON

        context['history_data_json'] = json.dumps(df_history_processed[['Date', 'Passagers_Reels']].to_dict(orient='records'))
    except FileNotFoundError:
        print(f"Avertissement: Le fichier d'historique {RAW_DATA_PATH} n'a pas été trouvé.")
    except Exception as e:
        print(f"Erreur lors du chargement des données historiques : {e}")


    return render(request, 'prediction_app/prediction_form.html', context)