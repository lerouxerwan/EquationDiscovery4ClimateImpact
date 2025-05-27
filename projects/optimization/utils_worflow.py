from typing import Any, Optional

from data.utils_dataset.dataset import Dataset
from data.utils_search.plot_diagnosis_search import plot_diagnosis_search
from emulator.pysr_emulator import PySREmulator
from emulator.utils_plots.utils_plots import plot_diagnosis_fit
from emulator_with_search.pysr_emulator_with_search import PySREmulatorWithSearch
from emulator_with_search.utils_attributes.utils_validation import get_X_and_y


def workflow(dataset: Dataset, params_emulator: dict[str, Any],
             params_search: Optional[dict[str, Any]] = None,
             show: bool = False) -> PySREmulatorWithSearch:
    """Workflow that fit an emulator to a dataset and generate diagnosis plots to assess fit quality
    This workflow takes as compulsory inputs: a dataset filename & a dictionary of parameters for the emulator
    An optional input is 'params_search' which gives some argument for hyperparameter search """
    # Load dataset
    (X_train, y_train, X_test, y_test, years_train, years_test,
        X_units, y_units, X_labels, y_labels, X_variables_names, y_variable_names,
        validation_mask) = dataset.values
    # Fit emulator
    if params_search is None:
        emulator  = PySREmulator(**params_emulator)
        X_train_train, y_train_train = get_X_and_y(X_train, y_train, validation_mask, validation_set=False)
        emulator.fit(X_train_train, y_train_train, variable_names=X_variables_names, X_units=X_units, y_units=y_units)
    else:
        emulator = PySREmulatorWithSearch(**params_emulator, **params_search)
        emulator.fit(X_train, y_train, variable_names=X_variables_names, X_units=X_units, y_units=y_units, validation_mask=validation_mask)
    #  Generate diagnosis plot for the fit
    plot_diagnosis_fit(emulator, dataset, show)
    #  Generate diagnosis plot for the search
    if params_search is not None:
        plot_diagnosis_search(dataset, emulator.search_experiment_, show)
    return emulator

