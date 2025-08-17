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

# Define vehicle speed limits (in km/hr)
# These values can be adjusted based on real-world constraints and data.
VEHICLE_SPEED_LIMITS = {
    "motorcycle": 90,  # Max avg speed for motorcycle
    "scooter": 60,     # Max avg speed for scooter
    "bicycle": 30      # Max avg speed for bicycle
}

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
                print(f"Best model ({model_name}) loaded from: {path}")
                best_model_found = True
                break
            except Exception as e:
                print(f"Error loading model from {path}: {e}")
    
    if not best_model_found:
        print(f"Error: No best model found with prefix '{BEST_MODEL_PREFIX}' in {MODEL_DIR}")
        return False
        
    return True

def get_prediction(user_input: dict):
    """
    gets real-time prediction and route visualization data based on user input.
    """
    if not load_inference_components():
        print("Failed to load inference components. Exiting.")
        return None, None

    # fetch real distance using OSRM
    pickup_lon = user_input['store_longitude']
    pickup_lat = user_input['store_latitude']
    drop_lon = user_input['drop_longitude']
    drop_lat = user_input['drop_latitude']

    real_distance = get_real_distance_from_osrm(pickup_lon, pickup_lat, drop_lon, drop_lat)
    if real_distance is None:
        print("Failed to get real distance. Prediction aborted.")
        return None, None
    user_input_data = user_input.copy()
    user_input_data['real_distance_km'] = real_distance

    # create DataFrame and preprocess
    expected_raw_columns = [
        'order_id', 'order_date', 'order_time', 'pickup_time',
        'store_latitude', 'store_longitude', 'drop_latitude', 'drop_longitude',
        'agent_age', 'agent_rating', 'traffic', 'weather', 'vehicle', 'area', 'category',
        'real_distance_km'
    ]
    input_df = pd.DataFrame([user_input_data])
    for col in expected_raw_columns:
        if col not in input_df.columns:
            input_df[col] = None # Ensure all expected columns are present, even if empty

    # convert time columns
    input_df['Order_Date'] = pd.to_datetime(input_df['order_date'])
    input_df['Order_Time'] = pd.to_datetime(input_df['order_time']).dt.time
    input_df['Pickup_Time'] = pd.to_datetime(input_df['pickup_time']).dt.time
    input_df['real_distance_km'] = input_df['real_distance_km'].astype(float)


    # select features relevant for the model
    # Ensure column names match the training data
    input_df = input_df.rename(columns={
        'store_latitude': 'Store_Latitude', 'store_longitude': 'Store_Longitude',
        'drop_latitude': 'Drop_Latitude', 'drop_longitude': 'Drop_Longitude',
        'agent_age': 'Agent_Age', 'agent_rating': 'Agent_Rating',
        'traffic': 'Traffic', 'weather': 'Weather', 'vehicle': 'Vehicle',
        'area': 'Area', 'category': 'Category',
    })

    # Apply preprocessing pipeline
    try:
        processed_input = _preprocessing_pipeline.transform(input_df.copy())
        print("Input preprocessed successfully.")
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return None, None

    # make prediction
    try:
        predicted_time = _model.predict(processed_input)[0]
        # ensure prediction is not negative
        predicted_time = max(0, predicted_time) 
        print(f"Raw Predicted Delivery Time: {predicted_time:.2f} minutes")

        # --- Apply Speed Limit Logic ---
        vehicle_type = user_input['vehicle']
        if vehicle_type in VEHICLE_SPEED_LIMITS and real_distance > 0:
            max_allowed_speed_kmph = VEHICLE_SPEED_LIMITS[vehicle_type]
            implied_avg_speed_kmph = (real_distance / predicted_time) * 60 if predicted_time > 0 else float('inf')

            if implied_avg_speed_kmph > max_allowed_speed_kmph:
                # Calculate the minimum time required to not exceed the max speed
                min_time_minutes = (real_distance / max_allowed_speed_kmph) * 60
                predicted_time = max(predicted_time, min_time_minutes) # Use the higher of predicted or min_time
                print(f"Adjusted predicted time for {vehicle_type} due to speed limit. New time: {predicted_time:.2f} minutes. Implied speed: {real_distance / (predicted_time / 60):.2f} km/hr (capped at {max_allowed_speed_kmph} km/hr)")
            else:
                print(f"Implied speed for {vehicle_type}: {implied_avg_speed_kmph:.2f} km/hr (within limits)")

    except Exception as e:
        print(f"Error during prediction: {e}")
        return None, None

    # prepare data for map visualization
    map_data = pd.DataFrame([{
        'Order_ID': user_input['order_id'],
        'Store_Latitude': user_input['store_latitude'],
        'Store_Longitude': user_input['store_longitude'],
        'Drop_Latitude': user_input['drop_latitude'],
        'Drop_Longitude': user_input['drop_longitude'],
        'Predicted_Delivery_Time': predicted_time,
        'Real_Distance_km': real_distance,
        'Vehicle': user_input['vehicle'],
        'Traffic': user_input['traffic'],
        'Weather': user_input['weather']
    }])
    
    return predicted_time, map_data

if __name__ == '__main__':
    # basic test
    sample_input = {
        'order_id': 'TEST_001',
        'order_date': '2023-01-01',
        'order_time': '10:00:00',
        'pickup_time': '10:15:00',
        'store_latitude': 12.9716,
        'store_longitude': 77.5946,
        'drop_latitude': 13.0000,
        'drop_longitude': 77.6000,
        'agent_age': 25,
        'agent_rating': 4.8,
        'traffic': 'Low',
        'weather': 'Sunny',
        'vehicle': 'motorcycle',
        'area': 'Urban',
        'category': 'Food'
    }
    pred_time, map_df = get_prediction(sample_input)
    if pred_time is not None:
        print(f"\nPredicted Time for sample input: {pred_time:.2f} minutes")
        print("Map Data:\n", map_df)
    else:
        print("\nPrediction failed for sample input.")