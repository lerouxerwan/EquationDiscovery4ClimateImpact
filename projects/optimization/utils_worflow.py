from typing import Any, Optional

from data.utils_dataset.utils_dataset import load_dataset
from data.utils_search.plot_diagnosis_search import plot_diagnosis_search
from emulator.pysr_emulator import PySREmulator
from emulator.utils_plots.utils_plots import plot_diagnosis_fit
from emulator_with_search.pysr_emulator_with_search import PySREmulatorWithSearch
from emulator_with_search.utils_attributes.utils_validation import get_X_and_y


def workflow(dataset_filename: str, params_emulator: dict[str, Any], params_search: Optional[dict[str, Any]] = None,
             show: bool = False) -> PySREmulatorWithSearch:
    """Workflow that fit an emulator to a dataset and generate diagnosis plots to assess fit quality
    This workflow takes as compulsory inputs: a dataset filename & a dictionary of parameters for the emulator
    An optional input is 'params_search' which gives some argument for hyperparameter search """
    # Load dataset
    (X_train, y_train, X_test, y_test, X_units, y_units, years_train, years_test, rcp_name_train, rcp_name_test,
     variable_names, target_label, validation_mask) = load_dataset(dataset_filename)
    # Fit emulator
    if params_search is None:
        emulator  = PySREmulator(**params_emulator)
        X_train_train, y_train_train = get_X_and_y(X_train, y_train, validation_mask, validation_set=False)
        emulator.fit(X_train_train, y_train_train, variable_names=variable_names, X_units=X_units, y_units=y_units)
    else:
        emulator = PySREmulatorWithSearch(**params_emulator, **params_search)
        emulator.fit(X_train, y_train, variable_names=variable_names, X_units=X_units, y_units=y_units, validation_mask=validation_mask)
    #  Generate diagnosis plot for the fit
    plot_diagnosis_fit(emulator, X_train, y_train, validation_mask, X_test, y_test, years_train, years_test, rcp_name_train,
                       rcp_name_test, target_label, show)
    #  Generate diagnosis plot for the search
    if params_search is not None:
        plot_diagnosis_search(emulator.search_experiment_, show)
    return emulator

