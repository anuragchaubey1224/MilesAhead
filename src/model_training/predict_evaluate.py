# predict_evaluate.py

import os
import sys
import pandas as pd
import joblib
import requests
import json
import time

# path setup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# module imports
sys.path.append(os.path.join(BASE_DIR, 'src', 'feature_engineering'))
sys.path.append(os.path.join(BASE_DIR, 'src', 'model_training'))
sys.path.append(os.path.join(BASE_DIR, 'src', 'map_components'))



from feature_extraction import TemporalFeatures, UseRealDistance, TimeTakenFeature, DropRawColumns
from outlier import OutlierRemover
from encoding import ColumnStandardizer, ColumnDropper, EncodingTransformer
from scaling import CustomScalerTransformer
from feature_selection import TreeFeatureSelector

# global paths
PREPROCESSING_PIPELINE_PATH = os.path.join(MODEL_DIR, 'preprocessing_pipeline.joblib')
BEST_MODEL_PREFIX = 'Best_Tuned'

# global cache
_preprocessing_pipeline = None
_model = None

# OSRM API configuration
OSRM_URL = "http://router.project-osrm.org/route/v1/driving/{},{};{},{}?overview=false"

def get_real_distance_from_osrm(pickup_lon, pickup_lat, drop_lon, drop_lat):
    """
    fetches the real driving distance from the OSRM API.
    Returns distance in kilometers.
    """
    try:
        url = OSRM_URL.format(pickup_lon, pickup_lat, drop_lon, drop_lat)
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        if data['code'] == 'Ok' and 'routes' in data and data['routes']:
            distance_meters = data['routes'][0]['distance']
            distance_km = distance_meters / 1000
            print(f"OSRM API call successful. Distance: {distance_km:.2f} km")
            return distance_km
        else:
            print(f"Error from OSRM API: {data.get('code', 'Unknown error')}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Failed to connect to OSRM server: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during OSRM call: {e}")
        return None

def load_inference_components():
    """Load the pre-trained preprocessing pipeline and best model."""
    global _preprocessing_pipeline, _model

    if _preprocessing_pipeline is not None and _model is not None:
        print("Inference components already loaded.")
        return True

    # load preprocessing pipeline
    try:
        _preprocessing_pipeline = joblib.load(PREPROCESSING_PIPELINE_PATH)
        print(f"Preprocessing pipeline loaded from: {PREPROCESSING_PIPELINE_PATH}")
    except FileNotFoundError:
        print(f"Error: Preprocessing pipeline not found at {PREPROCESSING_PIPELINE_PATH}")
        return False
    except Exception as e:
        print(f"Error loading preprocessing pipeline: {e}")
        return False

    # load best tuned model
    best_model_found = False
    for model_name in ["XGBoost", "RandomForest", "LightGBM"]:
        path = os.path.join(MODEL_DIR, f'{BEST_MODEL_PREFIX}_{model_name}_Model.joblib')
        if os.path.exists(path):
            try:
                _model = joblib.load(path)
                print(f"Best tuned model ({model_name}) loaded.")
                best_model_found = True
                break
            except Exception as e:
                print(f"Error loading {model_name} model: {e}")

    if not best_model_found:
        print("Error: No tuned model found.")
        _preprocessing_pipeline = None
        return False

    return True

def get_prediction(user_input_data: dict) -> tuple:
    """preprocess and predict delivery time for a single input."""
    if not load_inference_components():
        return None, None

    #  call OSRM API to get real distance
    print("Fetching real distance from OSRM...")
    real_distance = get_real_distance_from_osrm(
        user_input_data['store_longitude'],
        user_input_data['store_latitude'],
        user_input_data['drop_longitude'],
        user_input_data['drop_latitude']
    )
    
    if real_distance is None:
        print("Failed to get real distance. Prediction aborted.")
        return None, None
    
    user_input_data['real_distance_km'] = real_distance
    
    #  create DataFrame and preprocess
    expected_raw_columns = [
        'order_id', 'order_date', 'order_time', 'pickup_time',
        'store_latitude', 'store_longitude',
        'drop_latitude', 'drop_longitude',
        'agent_age', 'agent_rating',
        'traffic', 'weather', 'vehicle', 'area', 'category',
        'real_distance_km'
    ]
    
    input_df = pd.DataFrame([user_input_data])
    for col in expected_raw_columns:
        if col not in input_df.columns:
            input_df[col] = pd.NA

    input_df = input_df[expected_raw_columns]
    input_df.columns = input_df.columns.str.lower()
    
    # domain validation
    if pd.notna(input_df.at[0, 'agent_age']) and not 18 <= input_df.at[0, 'agent_age'] <= 60:
        print(f"Warning: Agent age ({input_df.at[0, 'agent_age']}) is outside 18–60.")
    if pd.notna(input_df.at[0, 'agent_rating']) and not 1 <= input_df.at[0, 'agent_rating'] <= 5:
        print(f"Warning: Agent rating ({input_df.at[0, 'agent_rating']}) is outside 1–5.")

    data_for_map_viz = input_df[['order_id', 'store_latitude', 'store_longitude', 'drop_latitude', 'drop_longitude', 'real_distance_km']].copy()

    try:
        processed_input = _preprocessing_pipeline.transform(input_df)
        predicted_time = float(_model.predict(processed_input)[0])
        data_for_map_viz['predicted_time'] = predicted_time
        return predicted_time, data_for_map_viz
    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback; traceback.print_exc()
        return None, None

if __name__ == "__main__":
    print("Running predict_evaluate.py in standalone mode.")
    if not load_inference_components():
        sys.exit(1)

    sample_user_input = {
        'order_id': 'PRED_001_JUL_24',
        'order_date': '2025-07-24',
        'order_time': '19:00:00',
        'pickup_time': '19:10:00',
        'store_latitude': 12.9716,
        'store_longitude': 77.5946,
        'drop_latitude': 12.9279,
        'drop_longitude': 77.6271,
        'agent_age': 28,
        'agent_rating': 4.7,
        'traffic': 'High',
        'weather': 'Cloudy',
        'vehicle': 'Motorcycle',
        'area': 'Urban',
        'category': 'Food'
    }
    
    predicted_delivery_time, map_data_df = get_prediction(sample_user_input)

    if predicted_delivery_time is not None:
        print(f"\nPredicted Delivery Time: {predicted_delivery_time:.2f} minutes")
        print(map_data_df.head())

    else:
        print("Prediction failed.")
