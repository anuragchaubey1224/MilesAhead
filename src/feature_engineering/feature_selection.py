# feature_selection.py  (tree-based top feature selection)

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline


class TreeFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Select important features using a tree-based estimator.
    Compatible with full preprocessing pipeline.
    """
    def __init__(self, estimator=None, n_features_to_select=None):
        # default estimator
        if estimator is None:
            self.estimator = RandomForestRegressor(random_state=42, n_jobs=-1)
        else:
            self.estimator = estimator
            # ensure parallel processing if RandomForestRegressor
            if isinstance(self.estimator, RandomForestRegressor) and getattr(self.estimator, 'n_jobs', None) is None:
                self.estimator.n_jobs = -1

        self.n_features_to_select = n_features_to_select
        self.features_to_keep = None

    def fit(self, X, y):
        if y is None:
            raise ValueError("Target `y` must be provided for feature selection.")

        # ensure DataFrame with lowercase columns
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        X.columns = X.columns.str.lower()

        # fit estimator
        self.estimator.fit(X, y)

        # feature importances
        importances = self.estimator.feature_importances_
        indices = np.argsort(importances)[::-1]

        # select top features
        if self.n_features_to_select:
            top_indices = indices[:self.n_features_to_select]
        else:
            mean_importance = np.mean(importances)
            top_indices = [i for i, imp in enumerate(importances) if imp > mean_importance] or [indices[0]]

        self.features_to_keep = X.columns[top_indices].tolist()
        return self

    def transform(self, X):
        if self.features_to_keep is None:
            raise RuntimeError("Must fit before transform.")
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        X.columns = X.columns.str.lower()
        return X[self.features_to_keep]

    def get_feature_names_out(self, input_features=None):
        if self.features_to_keep is None:
            raise RuntimeError("Must fit before getting feature names.")
        return self.features_to_keep


def get_feature_selection_pipeline(method='tree_based', **kwargs):
    """Return feature selection pipeline."""
    if method == 'tree_based':
        return Pipeline([
            ('tree_selector', TreeFeatureSelector(
                estimator=kwargs.get('estimator', RandomForestRegressor(random_state=42, n_jobs=-1)),
                n_features_to_select=kwargs.get('n_features_to_select', None)
            ))
        ])
    raise ValueError(f"Unsupported feature selection method: {method}")


if __name__ == "__main__":
    print("Feature selection module ready — tree-based method with compatibility.")
