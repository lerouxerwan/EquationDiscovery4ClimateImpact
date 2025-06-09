from collections import Counter

import numpy as np
import pytest
from sympy import Symbol

from emulator.utils_potential_contributions.utils_data_augmentation import apply_data_augmentation
from emulator.utils_potential_contributions.utils_weighted_loss import get_weights
from tests.emulator.utils_tests_emulator import load_pysr_emulator_for_test, \
    run_three_main_functions_with_one_feature, load_X_and_y_for_test


@pytest.mark.parametrize("threshold_for_model_selection", [1.0, 1.5, 2.0])
def test_threshold_for_model_selection(threshold_for_model_selection):
    emulator = load_pysr_emulator_for_test(threshold_for_model_selection=threshold_for_model_selection)
    run_three_main_functions_with_one_feature(emulator, fit_with_validation_mask=False)
    # For threshold=1.0, we check that equation with maximal complexity is selected (because threshold=1.0 selects
    # the equation that minimizes the loss, i.e. the equation with maximum complexity of the Pareto front)
    if threshold_for_model_selection == 1.0:
        maximum_complexity = emulator.equations_['complexity'].iloc[-1]
        assert emulator.selected_complexity == maximum_complexity

@pytest.mark.parametrize("threshold_for_model_selection", [0.5, 2])
def test_invalid_threshold_for_model_selection(threshold_for_model_selection):
    with pytest.raises(AssertionError):
        load_pysr_emulator_for_test(threshold_for_model_selection=threshold_for_model_selection)


@pytest.mark.repeat(2)
def test_deterministic_and_compute_loss():
    emulator = load_pysr_emulator_for_test()
    X, y = load_X_and_y_for_test()
    emulator.fit(X, y)
    # Assert that the fit of the emulator is deterministic
    np.testing.assert_almost_equal(float(sum(emulator.loss_list)), 35698077.642941765)

def test_loss():
    emulator = load_pysr_emulator_for_test()
    X, y = load_X_and_y_for_test()
    emulator.fit(X, y)
    # Assert that the method compute_loss of the emulator is consistent with the loss column
    for loss1, loss2 in zip(emulator.loss_list, emulator.compute_loss_list(X, y)):
        np.testing.assert_almost_equal(float(loss1), loss2)
    # Predict with indexes that do not exist anymore
    for removed_index in [6, 7, 8]:
        with pytest.raises(IndexError):
            emulator.predict(X, index=removed_index)


list_of_X_units_and_expected_variable_names = [
    (['m', 's', 'mol', 'K', 'A', 'kg', 'cd'], ['x1']),
    (['s', 'm', 'mol', 'K', 'A', 'kg', 'cd'], ['x2']),
    (['s', 'mol', 'm', 'K', 'A', 'kg', 'cd'], ['x3']),
]

@pytest.mark.parametrize("X_units_and_expected_variable_names", list_of_X_units_and_expected_variable_names)
def test_units(X_units_and_expected_variable_names):
    # We force dimensionless constants for the test, because otherwise any variable (with any unit)
    # could be used in the equation, as long as it is multiplied by a constant that map its unit to the expected unit
    emulator = load_pysr_emulator_for_test(dimensionless_constants_only=True)
    X_units, expected_variable_names = X_units_and_expected_variable_names
    y_units = ['m']
    nb_features = len(X_units)
    X, y = load_X_and_y_for_test(nb_features=nb_features)
    variable_names = [f'x{i+1}' for i in range(nb_features)]
    emulator.fit(X, y, variable_names=variable_names, X_units=X_units, y_units=y_units)
    selected_variable_names = set(emulator.selected_expr.atoms(Symbol))
    sorted_selected_variable_names = sorted([str(variable_name) for variable_name in set(selected_variable_names)])
    # The selected expression should only contain expected variable to agree with the unit of the target
    assert list(sorted_selected_variable_names) == expected_variable_names

def test_composed_units():
    emulator = load_pysr_emulator_for_test(dimensionless_constants_only=True)
    X_units = ['', 'yr', 's^-1', 'm/s']
    y_units = ['m']
    nb_features = len(X_units)
    X, y = load_X_and_y_for_test(nb_features=nb_features)
    variable_names = [f'x{i+1}' for i in range(nb_features)]
    emulator.fit(X, y, variable_names=variable_names, X_units=X_units, y_units=y_units)


@pytest.mark.parametrize("data_augmentation_ratio", [2, 3])
def test_data_augmentation(data_augmentation_ratio: int):
    X, y = load_X_and_y_for_test()
    X_augmented, y_augmented = apply_data_augmentation(X, y, data_augmentation_ratio, data_augmentation_sigma=1.0)
    assert len(X_augmented) == data_augmentation_ratio * len(X)
    assert len(y_augmented) == data_augmentation_ratio * len(y)
    # data augmentation of X (with added noise) must be located in the end
    assert X[0] == X_augmented[0]
    assert X[-1] != X_augmented[-1]
    # data augmentation of y should only contain copies of y (no noise)
    assert y[0] == y_augmented[0]
    assert y[-1] == y_augmented[-1]

@pytest.mark.parametrize("weighted_loss_ratio", [2, 10])
def test_weighted_loss(weighted_loss_ratio):
    X, y = load_X_and_y_for_test()
    weights = get_weights(y, weighted_loss_ratio)
    assert all([1. <= weight <= weighted_loss_ratio for weight in weights])
    # Largest weights
    assert weights[np.argmax(y)] == weighted_loss_ratio
    assert weights[np.argmin(y)] == weighted_loss_ratio
    # Smallest weight
    assert 1 <= Counter(list(weights))[1.0] <= 2

def test_weighted_loss_special_case():
    weighted_loss_ratio = 3
    assert [int(v) for v in get_weights(np.array([-2, -1, 0, 1, 2]), weighted_loss_ratio)] == [3, 2, 1, 2, 3]
    assert [int(v) for v in get_weights(np.array([0, -1, 2, 1, -2]), weighted_loss_ratio)] == [1, 2, 3, 2, 3]
    weighted_loss_ratio = 2
    assert [int(v) for v in get_weights(np.array([10, 11, 12, 13]), weighted_loss_ratio)] == [2, 1, 1, 2]
    assert [int(v) for v in get_weights(np.array([13, 10, 11, 12]), weighted_loss_ratio)] == [2, 2, 1, 1]

@pytest.mark.parametrize("tournament_selection_n_and_population_size", [(10, 11), (8, 10), (8, 11), (10, 9)])
def test_adapt_tournament_selection_n(tournament_selection_n_and_population_size):
    tournament_selection_n, population_size = tournament_selection_n_and_population_size
    X, y = load_X_and_y_for_test()
    emulator = load_pysr_emulator_for_test(tournament_selection_n=tournament_selection_n, population_size=population_size)
    emulator.fit(X, y)




