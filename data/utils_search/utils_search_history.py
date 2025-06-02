from copy import deepcopy

from data.utils_search.experiment import Experiment
from data.utils_search.utils_heredity_tree import get_parent


def get_search_history(experiment: Experiment) -> list[Experiment]:
    search_history = [experiment]
    experiment_parent = get_parent(experiment)
    while experiment_parent is not None:
        search_history.append(deepcopy(experiment_parent))
        experiment_parent = get_parent(experiment_parent)
    return search_history[::-1]