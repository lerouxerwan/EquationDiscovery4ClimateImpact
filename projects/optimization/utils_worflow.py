from emulator.utils_plots.utils_plots import plot_diagnosis_fit
from emulator_with_search.pysr_emulator_with_search import PySREmulatorWithSearch
from data.utils_search.search_experiment import SearchExperiment
from data.utils_search.utils_best_score import get_best_search_experiment
from data.utils_search.plot_diagnosis_search import plot_diagnosis_search
from data.utils_search.utils_heredity_tree import add_heredity_link
from data.utils_dataset.utils_dataset import load_dataset


def workflow(dataset_filename: str, search_path_to_start_from: str | bool = False, **params_emulator):
    """Workflow that fit an emulator with search to a dataset and generate diagnosis plots to assess fit quality
    This workflow takes as inputs:
    a dataset filename, some parameters for the emulator with search, and a search folder path for parent:
        -Its default value is False, and means that no other search folder are taken into account
        -if it is a string representing a path, we load the best params from the parent search folder path
        -if it is the string "best", we load the best params from the search folder path minimizing the validation loss
    """
    # Load dataset
    (X_train, y_train, X_test, y_test, X_units, y_units, years_train, years_test, rcp_name_train, rcp_name_test,
     variable_names, target_label, ind_validation) = load_dataset(dataset_filename)
    # Start optimization from a previous search experiment
    if search_path_to_start_from:
        assert isinstance(search_path_to_start_from, str)
        if search_path_to_start_from == 'best':
            search_experiment = get_best_search_experiment(X_train, y_train, ind_validation)
        else:
            search_experiment = SearchExperiment(search_path_to_start_from)
        params_emulator = {**search_experiment.best_params, **params_emulator}
    # Fit emulator with search
    emulator = PySREmulatorWithSearch(**params_emulator)
    emulator.fit(X_train, y_train, variable_names=variable_names, X_units=X_units, y_units=y_units,
                 ind_validation=ind_validation)
    #  Generate diagnosis plot
    plot_diagnosis_fit(emulator, X_train, y_train, X_test, y_test, years_train, years_test, rcp_name_train,
                       rcp_name_test, ind_validation, target_label, False)
    plot_diagnosis_search(emulator.search_experiment_, False)
    # Add a child/parent link if 'search_path_to_start_from' was used
    if search_path_to_start_from:
        add_heredity_link(emulator.search_experiment_, search_experiment)

