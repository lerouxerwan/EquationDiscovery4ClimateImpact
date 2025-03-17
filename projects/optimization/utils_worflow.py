from data.utils_dataset.utils_dataset import load_dataset
from data.utils_search.plot_diagnosis_search import plot_diagnosis_search
from emulator.utils_plots.utils_plots import plot_diagnosis_fit
from emulator_with_search.pysr_emulator_with_search import PySREmulatorWithSearch


def workflow(dataset_filename: str, **params_emulator) -> PySREmulatorWithSearch:
    """Workflow that fit an emulator with search to a dataset and generate diagnosis plots to assess fit quality
    This workflow takes as inputs: a dataset filename, some parameters for the emulator with search"""
    # Load dataset
    (X_train, y_train, X_test, y_test, X_units, y_units, years_train, years_test, rcp_name_train, rcp_name_test,
     variable_names, target_label, validation_mask) = load_dataset(dataset_filename)
    # Fit emulator with search
    emulator = PySREmulatorWithSearch(**params_emulator)
    emulator.fit(X_train, y_train, variable_names=variable_names, X_units=X_units, y_units=y_units,
                 validation_mask=validation_mask)
    #  Generate diagnosis plot
    plot_diagnosis_fit(emulator, X_train, y_train, X_test, y_test, years_train, years_test, rcp_name_train,
                       rcp_name_test, validation_mask, target_label, False)
    plot_diagnosis_search(emulator.search_experiment_, False)
    return emulator

