import math

import numpy as np

from emulator.emulator import Emulator


def compute_optimal_threshold(emulator: Emulator, X_validation: np.ndarray, y_train: np.ndarray) -> float:
    train_loss_list = emulator.loss_list
    train_loss_min = min(train_loss_list)
    validation_loss_list = emulator.compute_loss_list(X_validation, y_train)
    index_validation_loss_min = np.nanargmin(validation_loss_list)
    train_loss_for_optimal_equation = train_loss_list[index_validation_loss_min]
    optimal_threshold = train_loss_for_optimal_equation / train_loss_min
    # Rounding above (with the ceiling function) the threshold above some digits:
    # This is done to avoid issues for the custom selection
    # Otherwise due to rounding in the multiplication operation, the correct equation was sometimes not selected
    # (because its loss value was just above min_loss_value * threshold, due to small roundings)
    nb_digits_for_upper_rounding = 10
    scaling = 10**nb_digits_for_upper_rounding
    optimal_threshold = float(math.ceil(optimal_threshold * scaling)) / scaling
    return optimal_threshold
