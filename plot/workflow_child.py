from typing import Any, Optional

import numpy as np

from data.utils_dataset.dataset import Dataset
from data.utils_experiment.experiment import Experiment
from data.utils_experiment.utils_top_experiment import get_top_experiment
from data.utils_experiment.utils_heredity_tree import add_heredity_link
from plot.workflow import workflow
from utils.utils_log import log_info


def workflow_child(dataset: Dataset, search_path_to_start_from: str, model_selection: str,
                   params_search: Optional[dict[str, Any]] = None, show: bool = False):
    """Workflow that fit an emulator with search to a dataset and generate diagnosis plots to assess fit quality
    This workflow takes as inputs: a dataset filename, some parameters for the emulator with search
    and a search folder path for parent:
        -if it is the string "top", we load the top params from the search folder path minimizing the validation loss
        -if it is a string representing a path, we load the top params from the parent search folder path

    """
    # Start optimization from a previous search experiment
    search_experiment = load_search_experiment(search_path_to_start_from,
                                               dataset.X_train, dataset.y_train, dataset.validation_mask,
                                               model_selection)
    log_info(f'Load top emulator params from: {search_experiment}')
    emulator = workflow(dataset, search_experiment.top_params, params_search, show)
    # Add a child/parent link if 'search_path_to_start_from' was used
    if params_search is not None:
        add_heredity_link(emulator.experiment_, search_experiment)


def load_search_experiment(experiment_path: str, X_train: np.ndarray, y_train: np.ndarray, validation_mask: np.ndarray[bool], model_selection: str) -> Experiment:
    assert isinstance(experiment_path, str)
    if experiment_path == 'top':
        return get_top_experiment(X_train, y_train, validation_mask, model_selection)
    else:
        return Experiment(experiment_path, model_selection)
