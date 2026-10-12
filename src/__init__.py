"""
lifestyle-biometric-age-prediction

Machine learning pipeline for age inference from lifestyle and biometric data.
Predicts age using health behaviors, eating patterns, physical activity, and BMI.

Author: Fayez Ahmad Protik
"""

from .preprocessor import LifestyleDataPreprocessor
from .models import (
    train_linear_regression,
    train_decision_tree,
    train_mlp_neural_network,
    evaluate_regression_model,
)

__version__ = "1.0.0"
__author__ = "Fayez Ahmad Protik"

__all__ = [
    "LifestyleDataPreprocessor",
    "train_linear_regression",
    "train_decision_tree",
    "train_mlp_neural_network",
    "evaluate_regression_model",
]
