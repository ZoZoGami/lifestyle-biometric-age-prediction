"""
preprocessor.py — Feature Engineering Pipeline for Lifestyle & Biometric Age Prediction

Transforms raw obesity and lifestyle telemetry into scaled feature matrices:
    - Binary encoding of health behaviors (FAVC, SCC, SMOKE)
    - Ordinal encoding of eating habits (FAVC, FCVC, NCP)
    - One-hot encoding of categorical variables (Gender, CALC, MTRANS)
    - Feature creation from BMI calculation (Height, Weight)
    - StandardScaler on continuous biometric measurements

Author: Fayez Ahmad Protik
"""

from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


class LifestyleDataPreprocessor:
    """Preprocessor for obesity/lifestyle dataset age prediction."""

    def __init__(self):
        self.label_encoders = {}
        self.one_hot_encoder = None
        self.scaler = StandardScaler()
        self.numeric_features = ['Height', 'Weight', 'BMI', 'CH2O', 'FAF']
        self.is_fitted = False

    def fit_transform(self, df: pd.DataFrame, target_col: str = 'Age') -> Tuple[pd.DataFrame, pd.Series]:
        """Fit preprocessing transformers and return processed (X, y)."""
        data = df.copy()

        # Extract target
        y = data[target_col].copy()

        # Compute BMI from height and weight
        if 'Height' in data.columns and 'Weight' in data.columns:
            data['Height_m'] = data['Height']
            data['Weight_kg'] = data['Weight']
            data['BMI'] = data['Weight_kg'] / (data['Height_m'] ** 2)

        # Binary features to encode
        binary_cols = ['FAVC', 'SCC', 'SMOKE']
        for col in binary_cols:
            if col in data.columns:
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col].astype(str))
                self.label_encoders[col] = le

        # Multi-category features for one-hot
        cat_cols = ['Gender', 'CALC', 'MTRANS']
        cat_cols = [c for c in cat_cols if c in data.columns]

        # Drop non-predictive/target columns
        cols_to_drop = [target_col] + [c for c in cat_cols if c in data.columns]
        X = data.drop(columns=cols_to_drop)

        # One-hot encode
        if cat_cols:
            self.one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            encoded_cats = self.one_hot_encoder.fit_transform(X[cat_cols])
            cat_names = self.one_hot_encoder.get_feature_names_out(cat_cols)
            encoded_df = pd.DataFrame(encoded_cats, columns=cat_names, index=X.index)
            X = X.drop(columns=cat_cols).join(encoded_df)

        # Scale continuous features
        scale_cols = [c for c in self.numeric_features if c in X.columns]
        if scale_cols:
            X[scale_cols] = self.scaler.fit_transform(X[scale_cols])

        self.is_fitted = True
        return X, y
