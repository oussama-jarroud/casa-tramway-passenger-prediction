from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.contrib import messages
import pandas as pd
import joblib
import os
import json
from .services.data_processing import preprocess_data, load_events_holidays_data

# --- Chemins ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'saved_models')
RAW_DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'passengers_casatramway_raw.csv')
EVENTS_HOLIDAYS_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'events_holidays.csv')
MEDIA_ROOT_DIR = os.path.join(BASE_DIR, 'media')

# --- Chargement des événements/vacances ---
try:
    EVENTS_HOLIDAYS_DF = load_events_holidays_data(EVENTS_HOLIDAYS_PATH)
except Exception as e:
    EVENTS_HOLIDAYS_DF = pd.DataFrame()
    print(f"Avertissement : impossible de charger {EVENTS_HOLIDAYS_PATH} : {e}")

# --- Chargement des modèles ML ---
MODELS = {}
for name, filename in [('XGBoost','xgboost_model.pkl'),
                       ('Random Forest','random_forest_model.pkl'),
                       ('Linear Regression','linear_regression_model.pkl')]:
    try:
        MODELS[name] = joblib.load(os.path.join(MODEL_DIR, filename))
        print(f"{name} chargé avec succès")
    except Exception as e:
        print(f"Erreur chargement {name} : {e}")

# --- Features attendues ---
EXPECTED_FEATURES = [
    'Annee', 'Mois', 'Jour', 'Jour_Semaine', 'Est_Weekend',
    'Est_Jour_Ferie', 'Est_Vacances_Scolaires', 'Evenement_Special',
    'Temperature_Moyenne_C', 'Precipitations_mm',
    'Jour_Friday', 'Jour_Monday', 'Jour_Saturday', 'Jour_Sunday',
    'Jour_Thursday', 'Jour_Tuesday', 'Jour_Wednesday'
]

def predict_view(request):
    context = {}
    if request.method == 'POST' and request.FILES.get('csv_file'):
        uploaded_file = request.FILES['csv_file']
        fs = FileSystemStorage(location=MEDIA_ROOT_DIR)
        filename = fs.save(uploaded_file.name, uploaded_file)
        file_path = fs.path(filename)

        try:
            df_uploaded = pd.read_csv(file_path)
            df_processed = preprocess_data(df_uploaded.copy(), EVENTS_HOLIDAYS_DF.copy())
            df_processed['Date'] = pd.to_datetime(df_processed['Date'])

            model_name = request.POST.get('model_choice', 'XGBoost')
            model = MODELS.get(model_name)
            if not model:
                messages.error(request, f"Le modèle {model_name} n'est pas disponible.")
            else:
                df_features_for_prediction = df_processed[EXPECTED_FEATURES]
                predictions = model.predict(df_features_for_prediction)
                df_processed['Predictions'] = predictions.tolist()
                df_processed['Date'] = df_processed['Date'].dt.strftime('%Y-%m-%d')

                chart_data = df_processed[['Date', 'Passagers_Reels', 'Predictions']].to_dict(orient='records')

                context['model_used'] = model_name
                context['predictions_df'] = df_processed.to_html(classes='table table-striped', index=False)
                context['chart_data_json'] = json.dumps(chart_data)

                messages.success(request, f"✅ Prédiction terminée avec succès avec {model_name} !")

        except ValueError as e:
            messages.error(request, f"Erreur de données : {e}")
        except Exception as e:
            messages.error(request, f"Une erreur est survenue : {e}")
        finally:
            if os.path.exists(file_path):
                fs.delete(filename)

    # --- Charger les données historiques ---
    try:
        df_history_raw = pd.read_csv(RAW_DATA_PATH)
        df_history_processed = preprocess_data(df_history_raw.copy(), EVENTS_HOLIDAYS_DF.copy())
        if 'Nb_Passagers' in df_history_processed.columns:
            df_history_processed.rename(columns={'Nb_Passagers':'Passagers_Reels'}, inplace=True)
        df_history_processed['Date'] = pd.to_datetime(df_history_processed['Date']).dt.strftime('%Y-%m-%d')
        context['history_data_json'] = json.dumps(df_history_processed[['Date','Passagers_Reels']].to_dict(orient='records'))
    except Exception as e:
        print(f"Erreur chargement historique : {e}")

    return render(request, 'prediction_app/prediction_form.html', context)
