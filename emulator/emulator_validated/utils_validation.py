import numpy as np
from numpy import ndarray


def get_X_and_y(X: np.ndarray, y: np.ndarray, validation_mask: np.ndarray, validation_set: bool) -> tuple[np.ndarray, ndarray]:
    """Split X and y between train and validation.
    Returns X_train and y_train if validation_set=False other returns X_validation and y_validation"""
    return (X[validation_mask, :], y[validation_mask]) if validation_set else (X[~validation_mask, :], y[~validation_mask])


