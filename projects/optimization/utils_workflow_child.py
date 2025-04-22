from typing import Any, Optional

from data.utils_dataset.utils_dataset import load_dataset
from data.utils_search.search_experiment import SearchExperiment
from data.utils_search.utils_best_score import get_best_search_experiment
from data.utils_search.utils_heredity_tree import add_heredity_link
from projects.optimization.utils_worflow import workflow
from utils.utils_log import log_info


def workflow_child(dataset_filename: str, search_path_to_start_from: str, params_search: Optional[dict[str, Any]] = None,
                   show: bool = False):
    """Workflow that fit an emulator with search to a dataset and generate diagnosis plots to assess fit quality
    This workflow takes as inputs: a dataset filename, some parameters for the emulator with search
    and a search folder path for parent:
        -if it is the string "best", we load the best params from the search folder path minimizing the validation loss
        -if it is a string representing a path, we load the best params from the parent search folder path

    """
    # Load dataset
    (X_train, y_train, X_test, y_test, X_units, y_units, years_train, years_test, rcp_name_train, rcp_name_test,
     variable_names, target_label, validation_mask) = load_dataset(dataset_filename)
    # Start optimization from a previous search experiment
    search_experiment = load_search_experiment(search_path_to_start_from, X_train, y_train, validation_mask)
    log_info(f'Load best emulator params from: {search_experiment}')
    emulator = workflow(dataset_filename, search_experiment.best_params, params_search, show)
    # Add a child/parent link if 'search_path_to_start_from' was used
    if params_search is not None:
        add_heredity_link(emulator.search_experiment_, search_experiment)


def load_search_experiment(search_path_to_start_from, X_train, y_train, validation_mask):
    assert isinstance(search_path_to_start_from, str)
    if search_path_to_start_from == 'best':
        return get_best_search_experiment(X_train, y_train, validation_mask)
    else:
        return SearchExperiment(search_path_to_start_from)
