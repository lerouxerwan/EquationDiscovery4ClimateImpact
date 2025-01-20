import numpy as np
import pytest
from sympy import Symbol

from data.dataset.utils_dataset import load_dataset_ndarray
from tests.emulator.utils_tests_emulator import load_climate_impact_emulator_for_test, \
    run_three_main_functions, load_X_and_y_for_1D_test


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
    X, y = load_X_and_y_for_1D_test()
    emulator.fit(X, y)
    # Assert that the fit of the emulator is deterministic
    loss_list = emulator.equations_['loss'].values
    np.testing.assert_almost_equal(float(loss_list.sum()), 35698078.93297232)
    # Assert that the method compute_loss of the emulator work well
    for loss1, loss2 in zip(loss_list, emulator.compute_loss(X, y)):
        np.testing.assert_almost_equal(float(loss1), loss2, decimal=0)

@pytest.mark.xfail
def test_units_v1():
    """Test loading of the 7 units from the international system"""
    emulator = load_climate_impact_emulator_for_test()
    filename_dataset = 'units_v1.csv'
    X, y, _, _, X_units, y_units, _, _, variable_names, _, _ = load_dataset_ndarray(filename_dataset)
    assert X_units == ['m', 's', 'mol', 'K', 'A', 'kg', 'cd']
    assert y_units == ['m']
    emulator.fit(X, y, variable_names=variable_names, X_units=X_units, y_units=y_units)
    selected_variable_names = set(emulator.selected_expr.atoms(Symbol))
    sorted_selected_variable_names = sorted([str(variable_name) for variable_name in set(selected_variable_names)])
    # The selected expression should only contain the variable 'x1', because it has the same unit as the target
    assert list(sorted_selected_variable_names) == ['x1']


