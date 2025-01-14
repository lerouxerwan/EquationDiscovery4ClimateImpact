import numpy as np
import pytest

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
def test_climate_impact_emulator_deterministic():
    emulator = load_climate_impact_emulator_for_test()
    X, y = load_X_and_y_for_1D_test()
    emulator.fit(X, y)
    total_loss = float(emulator.equations_['loss'].values.sum())
    np.testing.assert_almost_equal(total_loss, 18557963.2727)