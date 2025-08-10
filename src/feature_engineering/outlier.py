# outlier.py component
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class OutlierRemover(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self  

    def transform(self, X):
        X = X.copy()

        print("\n--- Starting Outlier Removal ---")
        print(f"Initial data shape: {X.shape}")

        # domain based filtering (for common sense columns)
        if "agent_age" in X.columns:
            X = X[(X["agent_age"] >= 18) & (X["agent_age"] <= 50)]
            print(f"  -> After 'agent_age' filtering, shape: {X.shape}")
        
        if "agent_rating" in X.columns:
            X = X[(X["agent_rating"] >= 1) & (X["agent_rating"] <= 5)]
            print(f"  -> After 'agent_rating' filtering, shape: {X.shape}")

        for col in ["order_hour", "order_minute", "pickup_hour", "pickup_minute"]:
            if col in X.columns:
                if "hour" in col:
                    X = X[X[col].between(0, 23, inclusive='both')]
                else:
                    X = X[X[col].between(0, 59, inclusive='both')]
                print(f"  -> After '{col}' filtering, shape: {X.shape}")

        #  IQR-based filtering (for statistical outliers)
        def remove_iqr_outliers(df, column):
            # check if column exists or if the dataframe is already empty
            if column not in df.columns or df.empty:
                print(f"  - Skipping IQR for '{column}'. DataFrame is empty or column not found.")
                return df
            
            initial_shape = df.shape[0]
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            filtered_df = df[(df[column] >= lower) & (df[column] <= upper)]

            # print statement after executing
            print(f"  -> IQR filtering on '{column}': Initial rows: {initial_shape}, Final rows: {filtered_df.shape[0]}, Range: ({lower:.2f}, {upper:.2f})")
            
            return filtered_df

        for col in ["real_distance_km", "order_to_pickup_min"]:
            X = remove_iqr_outliers(X, col)

        print(f"--- Outlier Removal Completed. Final shape: {X.shape} ---\n")
        return X
