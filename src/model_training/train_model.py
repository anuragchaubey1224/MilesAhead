# train_model.py

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#  path setup 
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

SELECTED_FEATURES_PATH = os.path.join(OUTPUT_DIR, 'selected_features.csv')

#  metric functions 
def calculate_metrics(y_true, y_pred) -> Dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "R2": r2_score(y_true, y_pred),
    }

def get_cv_metrics(model, X, y, cv: int = 5) -> Dict[str, float]:
    mae = -cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error', n_jobs=-1)
    rmse = -cross_val_score(model, X, y, cv=cv, scoring='neg_root_mean_squared_error', n_jobs=-1)
    r2 = cross_val_score(model, X, y, cv=cv, scoring='r2', n_jobs=-1)

    return {
        "CV_MAE_Mean": mae.mean(), "CV_MAE_Std": mae.std(),
        "CV_RMSE_Mean": rmse.mean(), "CV_RMSE_Std": rmse.std(),
        "CV_R2_Mean": r2.mean(), "CV_R2_Std": r2.std()
    }

#  training function 
def train_multiple_models(
    df: pd.DataFrame,
    target_column: str = "delivery_time",
    test_size: float = 0.2,
    random_state: int = 42,
    do_cross_validation: bool = False,
    cv_folds: int = 5
) -> Dict[str, Dict]:
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataframe.")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # ensure all features are numeric (XGBoost )
    non_numeric_cols = X.select_dtypes(include=['object', 'datetime64[ns]', 'timedelta64[ns]']).columns
    if len(non_numeric_cols) > 0:
        print(f"Warning: Dropping non-numeric columns before training: {list(non_numeric_cols)}")
        X = X.drop(columns=non_numeric_cols)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(random_state=random_state, n_jobs=-1),
        "XGBoost": XGBRegressor(random_state=random_state, verbosity=0, n_jobs=-1)
    }

    results = {
        "trained_models": {},
        "performance_metrics": {},
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test
    }

    for name, model in models.items():
        print(f"\nTraining model: {name}")
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            test_metrics = calculate_metrics(y_test, y_pred)
            results["performance_metrics"][name] = test_metrics
            print(f"Test -> MAE: {test_metrics['MAE']:.2f}, RMSE: {test_metrics['RMSE']:.2f}, R²: {test_metrics['R2']:.4f}")

            if do_cross_validation:
                print(f"Performing {cv_folds}-Fold Cross-Validation for {name}...")
                cv_metrics = get_cv_metrics(model, X, y, cv=cv_folds)
                results["performance_metrics"][name].update(cv_metrics)
                print(
                    f"CV Mean -> MAE: {cv_metrics['CV_MAE_Mean']:.2f}, "
                    f"RMSE: {cv_metrics['CV_RMSE_Mean']:.2f}, R²: {cv_metrics['CV_R2_Mean']:.4f}"
                )
                print(
                    f"CV Std  -> MAE: {cv_metrics['CV_MAE_Std']:.2f}, "
                    f"RMSE: {cv_metrics['CV_RMSE_Std']:.2f}, R²: {cv_metrics['CV_R2_Std']:.4f}"
                )

            model_path = os.path.join(MODEL_DIR, f"{name}.pkl")
            joblib.dump(model, model_path)
            print(f"Model saved to: {model_path}")
        except Exception as e:
            print(f"Error training {name}: {e}")

        results["trained_models"][name] = model

    return results

#  tuning function 
def tune_and_save_best_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    best_model_name: str,
    random_state: int = 42
):
    print(f"\nStarting Hyperparameter tuning for the best model: {best_model_name}")

    final_model = None
    if best_model_name == "XGBoost":
        model = XGBRegressor(random_state=random_state, verbosity=0, n_jobs=-1)
        param_grid = {
            'n_estimators': [100, 200, 300, 500],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7, 9],
            'subsample': [0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.7, 0.8, 0.9, 1.0]
        }
    elif best_model_name == "RandomForest":
        model = RandomForestRegressor(random_state=random_state, n_jobs=-1)
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
    elif best_model_name == "LinearRegression":
        print("Linear Regression does not require hyperparameter tuning.")
        final_model = LinearRegression()
        final_model.fit(X_train, y_train)
    else:
        print(f"Tuning not configured for model: {best_model_name}")
        return

    if best_model_name in ["XGBoost", "RandomForest"]:
        try:
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                scoring='neg_root_mean_squared_error',
                cv=5,
                n_jobs=-1,
                verbose=1
            )
            grid_search.fit(X_train, y_train)
            final_model = grid_search.best_estimator_
            print(f"Best parameters for {best_model_name}: {grid_search.best_params_}")
        except Exception as e:
            print(f"Error during tuning {best_model_name}: {e}")
            return

    if final_model is None:
        print(f"No model trained/tuned for {best_model_name}")
        return

    y_pred_final = final_model.predict(X_test)
    final_metrics = calculate_metrics(y_test, y_pred_final)
    print(f"\nFinal tuned model performance ({best_model_name}) on test set:")
    print(f"MAE: {final_metrics['MAE']:.2f}, RMSE: {final_metrics['RMSE']:.2f}, R²: {final_metrics['R2']:.4f}")

    best_model_path = os.path.join(MODEL_DIR, f"Best_Tuned_{best_model_name}_Model.joblib")
    joblib.dump(final_model, best_model_path)
    print(f"Best tuned model saved to: {best_model_path}")

#  main execution 
if __name__ == "__main__":
    print("Starting model training pipeline...")

    if not os.path.exists(SELECTED_FEATURES_PATH):
        print(f"Error: selected features file not found at {SELECTED_FEATURES_PATH}")
        print("Run 'main_pipeline_feature_selection_test.py' first to generate this file.")
    else:
        df_selected_features = pd.read_csv(SELECTED_FEATURES_PATH)
        print(f"Loaded selected features data with shape: {df_selected_features.shape}")

        training_results = train_multiple_models(
            df=df_selected_features,
            target_column="delivery_time",
            do_cross_validation=True,
            cv_folds=5
        )

        # pick best model based on RMSE (CV RMSE if available)
        best_rmse = float('inf')
        best_model_name = None
        for name, metrics in training_results["performance_metrics"].items():
            current_rmse = metrics.get('CV_RMSE_Mean', metrics['RMSE'])
            if current_rmse < best_rmse:
                best_rmse = current_rmse
                best_model_name = name

        print(f"\nBest initial model: {best_model_name} with RMSE: {best_rmse:.2f}")

        tune_and_save_best_model(
            X_train=training_results["X_train"],
            y_train=training_results["y_train"],
            X_test=training_results["X_test"],
            y_test=training_results["y_test"],
            best_model_name=best_model_name
        )

    print("\nModel training pipeline completed!")
