from copy import deepcopy

from emulator_with_search.utils_search_experiment.search_experiment import SearchExperiment
from emulator_with_search.utils_search_experiment.utils_heredity_tree import get_parent


def get_search_history(search_experiment: SearchExperiment) -> list[SearchExperiment]:
    search_history = [search_experiment]
    search_experiment_parent = get_parent(search_experiment)
    while search_experiment_parent is not None:
        search_history.append(deepcopy(search_experiment_parent))
        search_experiment_parent = get_parent(search_experiment_parent)
    return search_history[::-1]