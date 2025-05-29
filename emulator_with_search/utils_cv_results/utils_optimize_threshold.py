import numpy as np

from emulator.pysr_emulator import PySREmulator


def compute_optimal_threshold(emulator: PySREmulator, X_validation: np.ndarray, y_train: np.ndarray) -> float:
    train_loss_list = emulator.loss_list
    train_loss_min = min(train_loss_list)
    validation_loss_list = emulator.compute_loss_list(X_validation, y_train)
    index_validation_loss_min = np.nanargmin(validation_loss_list)
    train_loss_for_optimal_equation = train_loss_list[index_validation_loss_min]
    optimal_threshold = train_loss_for_optimal_equation / train_loss_min
    return optimal_threshold
