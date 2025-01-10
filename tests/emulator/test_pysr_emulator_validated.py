from tests.emulator.utils_tests_emulator import load_pysr_emulator_validated_for_test, run_three_main_functions


def test_pysr_emulator_validated_for_one_hyperparameter():
    # Run some validation with the hyperparameter 'populations' that can have two values 10 or 20
    emulator = load_pysr_emulator_validated_for_test(param_grid={'populations': [10, 20]})
    run_three_main_functions(emulator)