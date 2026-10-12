"""
models.py — Predictive Models for Lifestyle & Biometric Age Inference

Regression architectures for predicting age from lifestyle and biometric data:
    - Linear Regression (OLS Baseline)
    - Decision Tree Regressor (Depth-constrained)
    - Multi-Layer Perceptron (MLP) Neural Network Regressor

Author: Fayez Ahmad Protik
"""

from typing import Dict, Tuple
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def train_linear_regression(X_train: np.ndarray, y_train: np.ndarray) -> LinearRegression:
    """Train standard OLS Linear Regression baseline."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_decision_tree(
    X_train: np.ndarray,
    y_train: np.ndarray,
    max_depth: int = 10,
    random_state: int = 42,
) -> DecisionTreeRegressor:
    """Train depth-regularized Decision Tree Regressor."""
    model = DecisionTreeRegressor(max_depth=max_depth, random_state=random_state)
    model.fit(X_train, y_train)
    return model


def train_mlp_neural_network(
    X_train: np.ndarray,
    y_train: np.ndarray,
    hidden_layer_sizes: Tuple[int, ...] = (128, 64),
    max_iter: int = 1000,
    random_state: int = 42,
) -> MLPRegressor:
    """Train Multi-Layer Perceptron neural network with backpropagation."""
    model = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation="relu",
        solver="adam",
        max_iter=max_iter,
        random_state=random_state,
        early_stopping=True,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_regression_model(model, X_test: np.ndarray, y_test: np.ndarray, model_name: str = "Model") -> Dict[str, float]:
    """Compute MSE, RMSE, MAE, and R2 score for a regression model."""
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return {
        "Model": model_name,
        "MSE": round(float(mse), 2),
        "RMSE": round(float(rmse), 2),
        "MAE": round(float(mae), 2),
        "R2_Score": round(float(r2), 4),
    }
