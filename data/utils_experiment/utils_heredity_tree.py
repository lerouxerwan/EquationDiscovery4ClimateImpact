from typing import Optional

from data.utils_experiment.experiment import Experiment
import os.path as op

"""Write text files"""

def add_heredity_link(experiment_child: Experiment, experiment_parent: Experiment) -> None:
    """Create files that create some heredity link between the child and the parent"""
    # For the parent, we add the experiment_path of the children to a list of children (in a text file)
    filepath = experiment_parent.filepath_children
    _write_experiment_path(filepath, experiment_child.experiment_path, "a" if op.exists(filepath) else "w")
    # For the children, we add the experiment_path of the parent (in a text file)
    filepath = experiment_child.filepath_parent
    assert not op.exists(filepath)
    _write_experiment_path(filepath, experiment_parent.experiment_path)

def _write_experiment_path(filepath: str, experiment_path: str, option: str = 'w') -> None:
    file = open(filepath, option)
    file.write(f'{experiment_path}\n')
    file.close()

"""Read text files"""

def get_children(experiment: Experiment) -> list[Experiment]:
    """Returns list of child search experiments, returns empty list if the experiment has no children"""
    return _get_experiments(experiment.filepath_children, experiment.model_selection)

def get_parent(experiment: Experiment) -> Optional[Experiment]:
    """Returns parent search experiment, returns None if the search experiment has no parent"""
    experiments_parents = _get_experiments(experiment.filepath_parent, experiment.model_selection)
    assert len(experiments_parents) <= 1
    return None if len(experiments_parents) == 0 else experiments_parents[0]

def _get_experiments(filepath: str, model_selection: str) -> list[Experiment]:
    if op.exists(filepath):
        file = open(filepath, 'r')
        experiments = [Experiment(experiment_path[:-1], model_selection) for experiment_path in file.readlines()]
        file.close()
        return experiments
    else:
        return []
