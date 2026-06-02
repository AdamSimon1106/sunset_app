from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class WeatherPreprocessor(BaseEstimator, TransformerMixin):
    """
    Real-world-friendly preprocessing:
    - keep only allowed raw feature columns
    - add missing indicators
    - impute with train medians
    - preserve fixed column order
    """

    def __init__(self, feature_columns: list[str], add_missing_indicators: bool = True):
        self.feature_columns = feature_columns
        self.add_missing_indicators = add_missing_indicators

    def fit(self, X: pd.DataFrame, y=None):
        X = X.copy()

        missing_cols = [c for c in self.feature_columns if c not in X.columns]
        if missing_cols:
            raise ValueError(f"Missing required input columns during fit: {missing_cols}")

        X = X[self.feature_columns].copy()

        self.output_columns_ = self.feature_columns.copy()

        if self.add_missing_indicators:
            self.missing_indicator_columns_ = [f"{c}_missing" for c in self.feature_columns]
            self.output_columns_.extend(self.missing_indicator_columns_)
        else:
            self.missing_indicator_columns_ = []

        self.train_medians_ = {}
        for col in self.feature_columns:
            self.train_medians_[col] = X[col].median()

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        for col in self.feature_columns:
            if col not in X.columns:
                X[col] = np.nan

        X = X[self.feature_columns].copy()

        if self.add_missing_indicators:
            for col in self.feature_columns:
                X[f"{col}_missing"] = X[col].isna().astype(int)

        for col in self.feature_columns:
            X[col] = X[col].fillna(self.train_medians_[col])

        return X[self.output_columns_].copy()