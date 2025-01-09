import pytest

from tests.pysr_emulator.utils_tests_pysr_emulator import load_X_and_y_for_test, load_pysr_emulator_for_test


@pytest.mark.parametrize("threshold", [1.0, 1.5, 2.0])
def test_threshold_for_best_model_selection(threshold: float):
    X, y = load_X_and_y_for_test()
    estimator = load_pysr_emulator_for_test(threshold_for_best_model_selection=threshold)
    estimator.fit(X, y)
    estimator.predict(X)
    # For threshold=1.0, we check that equation with maximal complexity is selected (because threshold=1.0 selects
    # the equation that minimizes the loss, i.e. the equation with maximum complexity of the Pareto front)
    if threshold == 1.0:
        selected_complexity = estimator.get_best()['complexity']
        maximum_complexity = estimator.equations_['complexity'].iloc[-1]
        assert selected_complexity == maximum_complexity

@pytest.mark.parametrize("threshold", [0.5, 2])
def test_invalid_threshold_for_best_model_selection(threshold: float):
    with pytest.raises(AssertionError):
        load_pysr_emulator_for_test(threshold_for_best_model_selection=threshold)