from data.utils_experiment.experiment import Experiment
from data.utils_experiment.utils_heredity_tree import add_heredity_link, get_parent, get_children
from data.utils_search.utils_non_default_params import get_non_default_params
from data.utils_experiment.utils_experiment_path import get_experiment_path
from emulator.emulator_validated_with_search.emulator_with_search import EmulatorValidatedWithSearch
from tests.data.utils_tests_dataset import load_X_and_y_and_validation_mask_for_test


def test_search_experiment_tree():
    X, y, validation_mask = load_X_and_y_and_validation_mask_for_test()
    # Parent search experiment
    params_emulator_parent = {'n_iter': 1}
    non_default_params_parent = get_non_default_params(EmulatorValidatedWithSearch(**params_emulator_parent))
    parent_search_experiment = Experiment(get_experiment_path(X, y, validation_mask, non_default_params_parent))
    # Child search experiment
    params_emulator_child = {**params_emulator_parent, **{'adaptive_parsimony_scaling':500.}}
    non_default_params_child = get_non_default_params(EmulatorValidatedWithSearch(**params_emulator_child))
    child_search_experiment = Experiment(get_experiment_path(X, y, validation_mask, non_default_params_child))
    # Add heredity link (create parent and children files)
    add_heredity_link(child_search_experiment, parent_search_experiment)
    # Test get functions for parent
    assert get_parent(parent_search_experiment) is None
    assert get_parent(child_search_experiment).experiment_path == parent_search_experiment.experiment_path
    # Test get functions for children
    assert len(get_children(child_search_experiment)) == 0
    children_search_experiments = get_children(parent_search_experiment)
    assert len(children_search_experiments) == 1
    assert children_search_experiments[0].experiment_path == child_search_experiment.experiment_path
    # Remove folders
    child_search_experiment.remove_folder()
    parent_search_experiment.remove_folder()
