from numpy import ndarray

def get_X_and_y(X: ndarray, y: ndarray, validation_mask: ndarray, validation_set: bool) -> tuple[ndarray, ndarray]:
    """Split X and y between train and validation.
    Returns X_train and y_train if validation_set=False other returns X_validation and y_validation"""
    return (X[validation_mask, :], y[validation_mask]) if validation_set else (X[~validation_mask, :], y[~validation_mask])


