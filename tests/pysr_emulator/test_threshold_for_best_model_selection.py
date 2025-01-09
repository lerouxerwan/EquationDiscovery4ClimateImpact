import pytest

from tests.pysr_emulator.utils_tests_pysr_emulator import load_pysr_emulator_for_test, \
    run_three_main_functions


@pytest.mark.parametrize("threshold_for_best_model_selection", [1.0, 1.5, 2.0])
def test_threshold_for_best_model_selection(threshold_for_best_model_selection):
    emulator = load_pysr_emulator_for_test(threshold_for_best_model_selection=threshold_for_best_model_selection)
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
        load_pysr_emulator_for_test(threshold_for_best_model_selection=threshold_for_best_model_selection)