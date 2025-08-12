# main_pipeline.py

import os
import sys
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline

# path setup for the main pipeline
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'delivery_data_final.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

#  module imports (for all pipeline components)
sys.path.append(os.path.join(BASE_DIR, 'src', 'feature_engineering'))

#  UseRealDistance transformer
from feature_extraction import (
    TemporalFeatures,
    UseRealDistance,
    TimeTakenFeature,
    DropRawColumns
)
from outlier import OutlierRemover
from encoding import ColumnStandardizer, ColumnDropper, EncodingTransformer
from scaling import CustomScalerTransformer
from feature_selection import TreeFeatureSelector

sys.path.append(os.path.join(BASE_DIR, 'src', 'model_training'))
from train_model import train_multiple_models, tune_and_save_best_model


def run_full_pipeline():
    """
    execute the full pipeline
    """
    # 1. Data Loading
    print(" loading cleaned delivery data...")
    if not os.path.exists(DATA_PATH):
        print(f" Error: Data file not found at {DATA_PATH}")
        return
        
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.lower()
    print(f"initial data loaded with shape: {df.shape}")

    # Validate presence of real_distance_km (required by updated feature extraction)
    if 'real_distance_km' not in df.columns:
        print(" Error: 'real_distance_km' column not found in the dataset.")
        print(" Please ensure your API-augmented dataset contains 'real_distance_km' column.")
        return

    # Store original coordinates for visualization later (if available)
    if all(col in df.columns for col in ['order_id', 'store_latitude', 'store_longitude', 'drop_latitude', 'drop_longitude']):
        initial_coordinates_df = df[['order_id', 'store_latitude', 'store_longitude', 'drop_latitude', 'drop_longitude']].copy()

    TARGET_COLUMN = 'delivery_time'
    if TARGET_COLUMN not in df.columns:
        print(f" Error: target column '{TARGET_COLUMN}' not found in data ")
        return

    y = df[TARGET_COLUMN]
    X = df.drop(columns=[TARGET_COLUMN])

    # pipeline components (instances)
    temporal_features_step = TemporalFeatures()
    realdistance_step = UseRealDistance()          
    time_taken_feature_step = TimeTakenFeature()
    drop_raw_columns_step = DropRawColumns()

    column_standardizer_step = ColumnStandardizer()
    column_dropper_step = ColumnDropper(columns_to_drop=['order_id'])
    encoding_transformer_step = EncodingTransformer()

    custom_scaler_transformer_step = CustomScalerTransformer()
    tree_feature_selector_step = TreeFeatureSelector(n_features_to_select=10)

    # 2. Feature Extraction
    print("\n running feature extraction pipeline...")
    X_transformed = temporal_features_step.fit_transform(X.copy())
    X_transformed = realdistance_step.fit_transform(X_transformed)   
    X_transformed = time_taken_feature_step.fit_transform(X_transformed)
    X_transformed = drop_raw_columns_step.fit_transform(X_transformed)

    feature_path = os.path.join(OUTPUT_DIR, 'feature_extraction.csv')
    pd.concat([X_transformed, y.reset_index(drop=True)], axis=1).to_csv(feature_path, index=False)
    print(f" feature extraction completed. Shape: {X_transformed.shape}. Saved to:\n   {feature_path}")
    
    # 3. Outlier Removal
    print("\n removing outliers...")
    outlier_remover = OutlierRemover()
    df_for_outliers = pd.concat([X_transformed, y.rename('temp_target_for_outliers')], axis=1)
    df_outlier_removed_full = outlier_remover.fit_transform(df_for_outliers)

    if 'temp_target_for_outliers' not in df_outlier_removed_full.columns:
        print(" Error: outlier removal step removed the temporary target column unexpectedly.")
        return

    y_outlier_removed = df_outlier_removed_full['temp_target_for_outliers'].rename(TARGET_COLUMN).reset_index(drop=True)
    X_outlier_removed = df_outlier_removed_full.drop(columns=['temp_target_for_outliers']).reset_index(drop=True)
    
    outlier_path = os.path.join(OUTPUT_DIR, 'outlier.csv')
    pd.concat([X_outlier_removed, y_outlier_removed], axis=1).to_csv(outlier_path, index=False)
    print(f" outliers removed. Shape: {X_outlier_removed.shape}. Saved to:\n   {outlier_path}")
    
    # 4. Encoding
    print("\n running encoding pipeline...")
    X_encoded_standardized = column_standardizer_step.fit_transform(X_outlier_removed.copy())
    X_encoded_dropped = column_dropper_step.fit_transform(X_encoded_standardized)
    X_encoded = encoding_transformer_step.fit_transform(X_encoded_dropped)

    encoded_path = os.path.join(OUTPUT_DIR, 'encoded.csv')
    pd.concat([X_encoded, y_outlier_removed], axis=1).to_csv(encoded_path, index=False)
    print(f" encoding completed. Shape: {X_encoded.shape}. Saved to:\n   {encoded_path}")
    
    # 5. Scaling
    print("\n running scaling pipeline...")
    X_scaled = custom_scaler_transformer_step.fit_transform(X_encoded)
    scaled_path = os.path.join(OUTPUT_DIR, "scaled.csv")
    pd.concat([X_scaled, y_outlier_removed], axis=1).to_csv(scaled_path, index=False)
    print(f" scaling completed. Shape: {X_scaled.shape}. Saved to:\n   {scaled_path}")

    # ensure numeric columns before feature selection
    print("\n checking for non-numeric columns before feature selection...")
    non_numeric_cols = X_scaled.select_dtypes(include=['object', 'datetime64[ns]', 'timedelta64[ns]']).columns.tolist()
    if non_numeric_cols:
        print(f" warning: found non-numeric columns: {non_numeric_cols}")
        X_scaled = X_scaled.drop(columns=non_numeric_cols)
        print(f"non-numeric columns dropped. New shape: {X_scaled.shape}")
    else:
        print(" all columns numeric, proceeding...")
    
    # 6. Feature Selection
    print("\n running feature selection...")
    X_selected = tree_feature_selector_step.fit_transform(X_scaled, y_outlier_removed)
    
    selected_path = os.path.join(OUTPUT_DIR, "selected_features.csv")
    pd.concat([X_selected, y_outlier_removed], axis=1).to_csv(selected_path, index=False)
    print(f" feature selection completed. Shape: {X_selected.shape}. Saved to:\n   {selected_path}")

    # 7. Model Training
    print("\n starting model training and tuning...")
    df_for_training = pd.read_csv(selected_path)

    initial_shape = df_for_training.shape
    df_for_training.dropna(inplace=True)
    if df_for_training.shape != initial_shape:
        print(f" dropped {initial_shape[0] - df_for_training.shape[0]} rows with NaN values")
    
    training_results = train_multiple_models(
        df=df_for_training,
        target_column="delivery_time",
        do_cross_validation=True,
        cv_folds=5
    )

    best_rmse, best_model_name = float('inf'), None
    for name, metrics in training_results["performance_metrics"].items():
        current_rmse = metrics.get('CV_RMSE_Mean', metrics['RMSE'])
        if current_rmse < best_rmse:
            best_rmse, best_model_name = current_rmse, name
    
    print(f"\n best initial model: {best_model_name} with RMSE: {best_rmse:.2f}")

    tune_and_save_best_model(
        X_train=training_results["X_train"],
        y_train=training_results["y_train"],
        X_test=training_results["X_test"],
        y_test=training_results["y_test"],
        best_model_name=best_model_name
    )

    # load tuned model
    best_model_path = os.path.join(MODEL_DIR, f"Best_Tuned_{best_model_name}_Model.joblib")
    if not os.path.exists(best_model_path):
        print(f" error: Best tuned model not found at {best_model_path}")
        return
    final_best_model = joblib.load(best_model_path)

    # 8. save full inference pipeline
    print("\n creating and saving the full inference pipeline...")
    preprocessing_pipeline = Pipeline(steps=[
        ('temporal', temporal_features_step),
        ('realdistance', realdistance_step),          # renamed step in pipeline
        ('timediff', time_taken_feature_step),
        ('standardize', column_standardizer_step),
        ('drop_id', column_dropper_step), 
        ('encode', encoding_transformer_step), 
        ('scale', custom_scaler_transformer_step),
        ('select_features', tree_feature_selector_step),
        ('drop_raw_for_inference', DropRawColumns())
    ])
    
    preprocessing_pipeline_path = os.path.join(MODEL_DIR, 'preprocessing_pipeline.joblib')
    joblib.dump(preprocessing_pipeline, preprocessing_pipeline_path)
    print(f" preprocessing pipeline saved to: {preprocessing_pipeline_path}")
    print("\n full pipeline executed successfully!")


if __name__ == "__main__":
    run_full_pipeline()
