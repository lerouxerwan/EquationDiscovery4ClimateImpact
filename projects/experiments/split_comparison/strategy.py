import os
import os.path as op
from enum import StrEnum

from numpy import ndarray

from data.utils_dataset.dataset import Dataset
from data.utils_dataset.validation_split import ValidationSplit
from emulator.emulator import Emulator
from emulator.emulator_validated import EmulatorValidated
from plot.utils_plot import plot_diagnosis


class Strategy(StrEnum):
    PYSR = "'best' fitted on train + validation"
    # Only the two following strategy change with the validation split
    TRUE = "ground truth value on validation"
    BEST = "prediction with 'best' model selection fitted on train"
    CUSTOM = "prediction with 'custom' model selection fitted on train (and optimized on validation)"

def get_y_test(dataset: Dataset, folder: str):
    """Compute predictions with default parameters"""
    with_validation = dataset.validation_split is not ValidationSplit.NONE
    emulator = EmulatorValidated() if with_validation else Emulator()
    emulator.fit(dataset.X_train, dataset.y_train, dataset.validation_mask, dataset.X_variables_names,
                 dataset.X_units, dataset.y_units)
    # Predict and plot with default model selection ('custom' for EmulatorValidated, 'best' for Emulator)
    y_test_predict_custom = emulator.predict(dataset.X_test)
    plot_emulator(dataset, emulator, folder)
    # Predict and plot with 'best'
    if with_validation:
        emulator.model_selection = 'best'
        emulator.threshold_for_model_selection = 1.5
        y_test_predict_best = emulator.predict(dataset.X_test)
        plot_emulator(dataset, emulator, folder)
    else:
        y_test_predict_best = y_test_predict_custom
    # infos = [f'${emulator.selected_expr}$', emulator.selected_complexity, emulator.selected_variable_names]
    return dataset.y_test, y_test_predict_custom, y_test_predict_best
    # return {
    #     Strategy.TRUE: dataset.y_test,
    #     Strategy.CUSTOM: y_predict_test_custom,
    #     Strategy.BEST:y_predict_test_best,
    # }


def plot_emulator(dataset, emulator, folder):
    plot_folder = op.join(folder, dataset.y_variable_names[0], emulator.model_selection)
    if not op.exists(plot_folder):
        os.makedirs(plot_folder)
    plot_diagnosis(emulator, dataset, show=False, plot_folder=plot_folder)


if __name__ == '__main__':
    print(Strategy.CUSTOM)