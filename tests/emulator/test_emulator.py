import numpy as np
import pytest
from sympy import Symbol

from data.utils_dataset.utils_validation import get_X_and_y
from emulator.emulator import Emulator
from emulator.emulator_with_search import EmulatorWithSearch
from tests.data.utils_tests_dataset import load_X_and_y_and_validation_mask_for_test, load_X_and_y_for_test


@pytest.mark.parametrize("with_validation_mask", [False, True])
def test_emulator_fit(with_validation_mask: bool):
    emulator = Emulator(niterations=1)
    X, y, validation_mask = load_X_and_y_and_validation_mask_for_test()
    if with_validation_mask:
        emulator.fit(X, y, validation_mask)
        X_train, y_train = get_X_and_y(X, y, validation_mask, validation_set=False)
        loss_list = emulator.compute_loss_list(X_train, y_train)
    else:
        emulator.fit(X, y)
        loss_list = emulator.compute_loss_list(X, y)
    # Ensure that the two loss_list are consistent when they are rounded
    # It is normal if it is  sometimes not perfectly not consistent with the predict method
    # See https://github.com/MilesCranmer/PySR/discussions/943 for more details on this issue
    for loss_from_dataframe, loss_computed in zip(emulator.loss_list, loss_list):
        np.testing.assert_almost_equal(float(loss_from_dataframe), loss_computed, decimal=0)
    emulator.experiment_.remove_folder()

@pytest.mark.repeat(2)
def test_deterministic_and_compute_loss():
    emulator = Emulator(niterations=1)
    X, y = load_X_and_y_for_test()
    emulator.fit(X, y)
    # Assert that the fit of the emulator is deterministic
    np.testing.assert_almost_equal(float(sum(emulator.loss_list)), 35698078.93297232)
    emulator.experiment_.remove_folder()


list_of_X_units_and_expected_variable_names = [
    (['m', 's', 'mol', 'K', 'A', 'kg', 'cd'], ['x1']),
    (['s', 'm', 'mol', 'K', 'A', 'kg', 'cd'], ['x2']),
    (['s', 'mol', 'm', 'K', 'A', 'kg', 'cd'], ['x3']),
]

@pytest.mark.parametrize("X_units_and_expected_variable_names", list_of_X_units_and_expected_variable_names)
def test_units(X_units_and_expected_variable_names):
    # We force dimensionless constants for the test, because otherwise any variable (with any unit)
    # could be used in the equation, as long as it is multiplied by a constant that map its unit to the expected unit
    emulator = Emulator(niterations=1, dimensionless_constants_only=True)
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
    emulator.experiment_.remove_folder()

def test_composed_units():
    emulator = Emulator(niterations=1, dimensionless_constants_only=True)
    X_units = ['', 'yr', 's^-1', 'm/s']
    y_units = ['m']
    nb_features = len(X_units)
    X, y = load_X_and_y_for_test(nb_features=nb_features)
    variable_names = [f'x{i+1}' for i in range(nb_features)]
    emulator.fit(X, y, variable_names=variable_names, X_units=X_units, y_units=y_units)
    emulator.experiment_.remove_folder()


def test_save_then_load():
    # this test will crash if some fit attributes, such as 'feature_names_in_' are not set after loading
    X, y = load_X_and_y_for_test()
    emulator = Emulator(niterations=1)
    emulator.fit(X, y)
    emulator.predict(X)
    emulator = Emulator(niterations=1)
    emulator.fit(X, y)
    emulator.predict(X)
    emulator.experiment_.remove_folder()



