import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

#  temporal feature extraction 
class TemporalFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # parse datetime columns safely
        order_time = pd.to_datetime(X['order_time'], errors='coerce').dt.strftime('%H:%M:%S')
        pickup_time = pd.to_datetime(X['pickup_time'], errors='coerce').dt.strftime('%H:%M:%S')

        X['order_datetime'] = pd.to_datetime(
            X['order_date'] + ' ' + order_time, errors='coerce'
        )
        X['pickup_datetime'] = pd.to_datetime(
            X['order_date'] + ' ' + pickup_time, errors='coerce'
        )

        # extract useful features 
        X['order_hour'] = X['order_datetime'].dt.hour
        X['order_minute'] = X['order_datetime'].dt.minute
        X['order_dayofweek'] = X['order_datetime'].dt.dayofweek
        X['is_weekend'] = X['order_dayofweek'].isin([5, 6]).astype(int)
        X['is_peakhour'] = X['order_hour'].isin([8, 9, 18, 19]).astype(int)
        X['pickup_hour'] = X['pickup_datetime'].dt.hour
        X['pickup_minute'] = X['pickup_datetime'].dt.minute

        return X

#  using real distance for model
class UseRealDistance(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        # Ensure real_distance_km exists
        if 'real_distance_km' not in X.columns:
            raise ValueError("Missing column: real_distance_km in dataset")
        return self

    def transform(self, X):
        X = X.copy()
        X['real_distance_km'] = pd.to_numeric(X['real_distance_km'], errors='coerce')
        return X

# time taken features
class TimeTakenFeature(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['order_to_pickup_min'] = (
            X['pickup_datetime'] - X['order_datetime']
        ).dt.total_seconds() / 60.0
        return X

# drop raw columns 
class DropRawColumns(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        drop_cols = [
            'order_date', 'order_time', 'pickup_time',
            'store_latitude', 'store_longitude',
            'drop_latitude', 'drop_longitude',
            'order_datetime', 'pickup_datetime'
        ]
        return X.drop(columns=[col for col in drop_cols if col in X.columns])

# feature extraction pipeline
feature_pipeline = Pipeline([
    ('temporal', TemporalFeatures()),
    ('realdistance', UseRealDistance()),
    ('timediff', TimeTakenFeature()),
    ('dropcols', DropRawColumns())
])

# main pipeline entry point
if __name__ == "__main__":
    print("Feature extraction module updated to use real_distance_km")
