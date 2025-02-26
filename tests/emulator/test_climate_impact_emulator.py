import numpy as np
import pytest
from sklearn.utils import check_random_state
from sympy import Symbol

from data.utils_dataset import load_dataset_ndarray
from emulator.utils_hyperparameter_search.utils_feature_selection import get_selection_mask
from emulator.utils_hyperparameter_search.utils_validation import compute_ind_validation, get_X_and_y
from tests.emulator.utils_tests_emulator import load_climate_impact_emulator_for_test, \
    run_three_main_functions, load_X_and_y_for_test
from utils.utils_run import random_seed



@pytest.mark.parametrize("threshold_for_best_model_selection", [1.0, 1.5, 2.0])
def test_threshold_for_best_model_selection(threshold_for_best_model_selection):
    emulator = load_climate_impact_emulator_for_test(threshold_for_best_model_selection=threshold_for_best_model_selection)
    run_three_main_functions(emulator)
    # For threshold=1.0, we check that equation with maximal complexity is selected (because threshold=1.0 selects
    # the equation that minimizes the loss, i.e. the equation with maximum complexity of the Pareto front)
    if threshold_for_best_model_selection == 1.0:
        selected_complexity = emulator.get_best()['complexity']
        maximum_complexity = emulator.equations_['complexity'].iloc[-1]
        assert selected_complexity == maximum_complexity

@pytest.mark.parametrize("threshold_for_best_model_selection", [0.5, 2])
def test_invalid_threshold_for_best_model_selection(threshold_for_best_model_selection):
    with pytest.raises(AssertionError):
        load_climate_impact_emulator_for_test(threshold_for_best_model_selection=threshold_for_best_model_selection)


@pytest.mark.repeat(2)
def test_deterministic_and_compute_loss():
    emulator = load_climate_impact_emulator_for_test()
    X, y = load_X_and_y_for_test()
    emulator.fit(X, y)
    # Assert that the fit of the emulator is deterministic
    np.testing.assert_almost_equal(float(sum(emulator.loss_list)), 35698078.93297232)

def test_loss():
    emulator = load_climate_impact_emulator_for_test()
    X, y = load_X_and_y_for_test()
    emulator.fit(X, y)
    # Assert that the method compute_loss of the emulator work well
    for loss1, loss2 in zip(emulator.loss_list, emulator.compute_loss(X, y)):
        np.testing.assert_almost_equal(float(loss1), loss2, decimal=0)

list_of_X_units_and_expected_variable_names = [
    (['m', 's', 'mol', 'K', 'A', 'kg', 'cd'], ['x1']),
    (['s', 'm', 'mol', 'K', 'A', 'kg', 'cd'], ['x2']),
    (['s', 'mol', 'm', 'K', 'A', 'kg', 'cd'], ['x3']),
]

@pytest.mark.parametrize("X_units_and_expected_variable_names", list_of_X_units_and_expected_variable_names)
def test_units(X_units_and_expected_variable_names):
    # We force dimensionless constants for the test, because otherwise any variable (with any unit)
    # could be used in the equation, as long as it is multiplied by a constant that map its unit to the expected unit
    emulator = load_climate_impact_emulator_for_test(dimensionless_constants_only=True)
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
    emulator = load_climate_impact_emulator_for_test(dimensionless_constants_only=True)
    X_units = ['', 'yr', 's^-1', 'm/s']
    y_units = ['m']
    nb_features = len(X_units)
    X, y = load_X_and_y_for_test(nb_features=nb_features)
    variable_names = [f'x{i+1}' for i in range(nb_features)]
    emulator.fit(X, y, variable_names=variable_names, X_units=X_units, y_units=y_units)


list_of_feature_selection_name_and_selected_features = [
    ('PySRDefault', ['Max_VEddyDiff_MAM', 'Mean_SSS_MAM', 'Mean_MLD_MAM', 'Max_MLD_DJF']),
]

@pytest.mark.parametrize("feature_selection_name_and_selected_features", list_of_feature_selection_name_and_selected_features)
def test_feature_selection(feature_selection_name_and_selected_features):
    feature_selection_name, selected_features_expected = feature_selection_name_and_selected_features
    filename = r"NPP_season.csv"
    X, y, _, _, _, _, _, _, _, _, variable_names, _, nb_historical_years = load_dataset_ndarray(filename)
    nb_features = 4
    random_state = check_random_state(random_seed)
    _ = random_state.randint(0, 2 ** 31 - 1)  # To have exactly the same random state as during the fit function

    ind_validation = compute_ind_validation(len(y), 0.3, nb_historical_years)
    X_train_train, y_train_train = get_X_and_y(X, y, ind_validation, validation_set=False)
    selection_mask = get_selection_mask(X_train_train, y_train_train, nb_features, feature_selection_name, variable_names, random_state)
    assert sum(selection_mask) == nb_features
    # Check that selected features are as expected
    assert list(variable_names[selection_mask]) == list(selected_features_expected)



