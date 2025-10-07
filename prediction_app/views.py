# prediction_app/views.py
from django.shortcuts import render
from django.http import HttpResponse # Pour les réponses simples, utile pour les erreurs
from django.core.files.storage import FileSystemStorage
import pandas as pd
import joblib
import os
import json # Pour passer les données aux templates en JSON

# Importe tes fonctions de preprocessing
# Assure-toi que le chemin d'accès est correct par rapport à views.py
# Il faut que python puisse trouver ton module 'src'
# Une astuce est d'ajouter le dossier racine du projet au PYTHONPATH
# ou de copier src/data_processing.py dans prediction_app/services/
# Pour simplifier, nous allons le charger directement ici
from .services.data_processing import preprocess_data, load_events_holidays_data # Crée ce fichier

# Définis les chemins vers tes modèles et tes données brutes/événements
# Ces chemins sont relatifs au dossier racine du projet Django
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'saved_models')
RAW_DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'passengers_casatramway_raw.csv') # Données historiques pour référence
EVENTS_HOLIDAYS_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'events_holidays.csv')

# Charge les données d'événements et jours fériés une seule fois au démarrage de l'app
# ou au premier appel si elles sont petites
try:
    EVENTS_HOLIDAYS_DF = load_events_holidays_data(EVENTS_HOLIDAYS_PATH)
except FileNotFoundError:
    EVENTS_HOLIDAYS_DF = pd.DataFrame() # DataFrame vide si le fichier n'existe pas
    print(f"Attention: Le fichier {EVENTS_HOLIDAYS_PATH} n'a pas été trouvé. Les événements/vacances ne seront pas inclus.")

# Charger les modèles une seule fois pour ne pas les recharger à chaque requête
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


def predict_view(request):
    context = {}
    if request.method == 'POST' and request.FILES.get('csv_file'):
        uploaded_file = request.FILES['csv_file']
        fs = FileSystemStorage(location=os.path.join(BASE_DIR, 'media')) # Sauvegarde dans le dossier media
        filename = fs.save(uploaded_file.name, uploaded_file)
        file_path = fs.path(filename)

        try:
            # 1. Charger le CSV uploadé
            df_uploaded = pd.read_csv(file_path)

            # 2. Appliquer le preprocessing
            # Nous allons utiliser une version modifiée de preprocess_data
            # qui prend un DataFrame en entrée
            df_processed = preprocess_data(df_uploaded.copy(), EVENTS_HOLIDAYS_DF.copy())

            # Vérifier si la colonne cible existe pour les données historiques (si tu veux l'afficher)
            # et renommer 'Nb_Passagers' pour la clarté si elle existe
            if 'Nb_Passagers' in df_processed.columns:
                df_processed.rename(columns={'Nb_Passagers': 'Passagers_Reels'}, inplace=True)
                df_processed['Passagers_Reels'] = df_processed['Passagers_Reels'].astype(float) # S'assurer du type numérique

            # 3. Sélectionner le modèle et faire la prédiction
            model_name = request.POST.get('model_choice', 'XGBoost') # Récupérer le choix du formulaire
            model = MODELS.get(model_name)

            if model is None:
                context['error'] = f"Le modèle '{model_name}' n'est pas disponible ou n'a pas pu être chargé."
                return render(request, 'prediction_app/prediction_form.html', context)

            # Assure-toi que df_processed contient EXACTEMENT les mêmes colonnes que lors de l'entraînement du modèle
            # et dans le même ordre. Cela peut être complexe et nécessiter une étape de validation/réarrangement ici.
            # Pour simplifier, supposons que c'est le cas pour l'instant.
            # Une meilleure approche serait de sauvegarder la liste des colonnes du training set.
            
            # Pour un exemple simple, créons un dataframe d'entrée pour la prédiction
            # en supprimant les colonnes qui ne sont pas des features
            # et s'assurer que les colonnes catégorielles sont traitées comme lors de l'entraînement
            features = [col for col in df_processed.columns if col not in ['Date', 'Passagers_Reels']]
            
            # Attention: Si tu as fait du One-Hot Encoding ou Scaler, il faut les appliquer ici aussi!
            # C'est pourquoi il est vital d'avoir un pipeline complet pour le preprocessing.
            
            predictions = model.predict(df_processed[features])
            df_processed['Predictions'] = predictions.tolist()


            # 4. Préparer les données pour le template (JSON)
            # Convertir les dates en string pour la sérialisation JSON
            df_processed['Date'] = df_processed['Date'].dt.strftime('%Y-%m-%d')

            # Combinaison des données réelles et prédites pour le graphique
            chart_data = df_processed[['Date', 'Passagers_Reels', 'Predictions']].to_dict(orient='records')
            
            context = {
                'model_used': model_name,
                'predictions_df': df_processed.to_html(classes='table table-striped', index=False), # Tableau HTML
                'chart_data_json': json.dumps(chart_data) # Données pour Plotly.js
            }

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
        df_history_processed = preprocess_data(df_history_raw.copy(), EVENTS_HOLIDAYS_DF.copy())
        if 'Nb_Passagers' in df_history_processed.columns:
             df_history_processed.rename(columns={'Nb_Passagers': 'Passagers_Reels'}, inplace=True)
        df_history_processed['Date'] = df_history_processed['Date'].dt.strftime('%Y-%m-%d')
        context['history_data_json'] = json.dumps(df_history_processed[['Date', 'Passagers_Reels']].to_dict(orient='records'))
    except FileNotFoundError:
        print(f"Avertissement: Le fichier d'historique {RAW_DATA_PATH} n'a pas été trouvé.")
    except Exception as e:
        print(f"Erreur lors du chargement des données historiques : {e}")

    return render(request, 'prediction_app/prediction_form.html', context)