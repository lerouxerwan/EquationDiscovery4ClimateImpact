from data.utils_search.utils_heredity_tree import add_heredity_link, get_parent, get_children
from data.utils_search.utils_search_path import get_non_default_params
from emulator_with_search.pysr_emulator_with_search import PySREmulatorWithSearch
from tests.data.utils_tests_dataset import load_X_and_y_and_validation_mask_for_test


def test_search_experiment_tree():
    # Fit one emulator_parent and one emulator_child
    X, y, validation_mask = load_X_and_y_and_validation_mask_for_test()
    params_emulator_parent = {'n_iter': 1}
    emulator_parent = PySREmulatorWithSearch(**params_emulator_parent)
    emulator_parent_search_experiment = emulator_parent.compute_emulator_search_experiment(X, y, validation_mask, get_non_default_params(emulator_parent))
    params_emulator_child = {**params_emulator_parent, **{'adaptive_parsimony_scaling':500.}}
    emulator_child = PySREmulatorWithSearch(**params_emulator_child)
    emulator_child_search_experiment = emulator_child.compute_emulator_search_experiment(X, y, validation_mask, get_non_default_params(emulator_child))
    # Add heredity link (create parent and children files)
    add_heredity_link(emulator_child_search_experiment, emulator_parent_search_experiment)
    # Test get functions for parent
    assert get_parent(emulator_parent_search_experiment) is None
    assert get_parent(emulator_child_search_experiment).search_path == emulator_parent_search_experiment.search_path
    # Test get functions for children
    assert len(get_children(emulator_child_search_experiment)) == 0
    children_search_experiments = get_children(emulator_parent_search_experiment)
    assert len(children_search_experiments) == 1
    assert children_search_experiments[0].search_path == emulator_child_search_experiment.search_path
    # Remove folders
    emulator_child_search_experiment.remove_folder()
    emulator_parent_search_experiment.remove_folder()
