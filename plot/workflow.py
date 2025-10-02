from typing import Any, Optional

from data.utils_dataset.dataset import Dataset
from emulator.emulator import Emulator
from emulator.emulator_with_search import EmulatorWithSearch
from plot.utils_metric.metric import Metric
from plot.plot_diagnosis import plot_diagnosis


def workflow(dataset: Dataset, params_emulator: dict[str, Any],
             params_search: Optional[dict[str, Any]] = None,
             show: bool = False, plot_folder: Optional[str] = None) -> EmulatorWithSearch:
    """Workflow that fit an emulator to a dataset and generate diagnosis plots to assess fit quality
    This workflow takes as compulsory inputs: a dataset filename & a dictionary of parameters for the emulator
    An optional input is 'params_search' which gives some argument for hyperparameter search """
    # Fit emulator
    if params_search is None:
        emulator  = Emulator(**params_emulator)
    else:
        emulator = EmulatorWithSearch(**params_emulator, **params_search)
    fit(emulator, dataset)
    #  Generate diagnosis plot for the fit
    plot_diagnosis(emulator, dataset, show, plot_folder)
    return emulator


def fit(emulator: Emulator, dataset: Dataset) -> None:
    emulator.fit(dataset.X_train, dataset.y_train, dataset.validation_mask,
                 dataset.X_variable_names, dataset.X_units, dataset.y_units)

def compute_loss_test(emulator: Emulator, dataset: Dataset, metric: Metric) -> float:
    return emulator.compute_loss(dataset.X_test, dataset.y_test, metric)

