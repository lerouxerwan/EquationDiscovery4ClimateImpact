from typing import Optional

from emulator_with_search.utils_search_experiment.search_experiment import SearchExperiment
import os.path as op

"""Write text files"""

def add_heredity_link(search_experiment_child: SearchExperiment, search_experiment_parent: SearchExperiment) -> None:
    """Create files that create some heredity link between the child and the parent"""
    # For the parent, we add the search_path of the children to a list of children (in a text file)
    filepath = search_experiment_parent.filepath_children
    _write_search_path(filepath, search_experiment_child.search_path, "a" if op.exists(filepath) else "w")
    # For the children, we add the search_path of the parent (in a text file)
    filepath = search_experiment_child.filepath_parent
    assert not op.exists(filepath)
    _write_search_path(filepath, search_experiment_parent.search_path)

def _write_search_path(filepath: str, search_path: str, option: str = 'w') -> None:
    file = open(filepath, option)
    file.write(f'{search_path}\n')
    file.close()

"""Read text files"""

def get_children(search_experiment: SearchExperiment) -> list[SearchExperiment]:
    """Returns list of child search experiments, returns empty list if the search_experiment has no children"""
    return _get_search_experiments(search_experiment.filepath_children)

def get_parent(search_experiment: SearchExperiment) -> Optional[SearchExperiment]:
    """Returns parent search experiment, returns None if the search experiment has no parent"""
    search_experiments_parents = _get_search_experiments(search_experiment.filepath_parent)
    assert len(search_experiments_parents) <= 1
    return None if len(search_experiments_parents) == 0 else search_experiments_parents[0]

def _get_search_experiments(filepath: str) -> list[SearchExperiment]:
    if op.exists(filepath):
        file = open(filepath, 'r')
        search_experiments = [SearchExperiment(search_path[:-1]) for search_path in file.readlines()]
        file.close()
        return search_experiments
    else:
        return []
