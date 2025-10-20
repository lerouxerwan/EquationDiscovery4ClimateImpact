import warnings

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
    emulator.remove_folder()

@pytest.mark.repeat(2)
def test_deterministic_and_compute_loss():
    emulator = Emulator(niterations=1)
    X, y = load_X_and_y_for_test()
    emulator.fit(X, y)
    # Assert that the fit of the emulator is deterministic
    np.testing.assert_almost_equal(float(sum(emulator.loss_list)), 8511.4327203031)
    emulator.remove_folder()


def test_model_selection_validated():
    emulator = Emulator(niterations=1, model_selection='validated')
    X, y = load_X_and_y_for_test()
    emulator.fit(X, y)
    # Assert that the fit of the emulator is the same as with the model selection 'best'
    np.testing.assert_almost_equal(float(sum(emulator.loss_list)), 8511.4327203031)
    emulator.remove_folder()


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
    emulator.remove_folder()

def test_composed_units():
    emulator = Emulator(niterations=1, dimensionless_constants_only=True)
    X_units = ['', 'yr', 's^-1', 'm/s']
    y_units = ['m']
    nb_features = len(X_units)
    X, y = load_X_and_y_for_test(nb_features=nb_features)
    variable_names = [f'x{i+1}' for i in range(nb_features)]
    emulator.fit(X, y, variable_names=variable_names, X_units=X_units, y_units=y_units)
    emulator.remove_folder()


def test_save_then_load():
    # this test will crash if some fit attributes, such as 'feature_names_in_' are not set after loading
    X, y = load_X_and_y_for_test()
    emulator = Emulator(niterations=1)
    emulator.fit(X, y)
    emulator.predict(X)
    emulator = Emulator(niterations=1)
    emulator.fit(X, y)
    emulator.predict(X)
    emulator.remove_folder()

def test_warm_start():
    X, y = load_X_and_y_for_test()
    emulator = Emulator(niterations=20)
    emulator.fit(X, y)
    complexity_to_loss = dict(zip(emulator.complexity_list, emulator.loss_list))
    emulator.warm_start = True
    emulator.niterations = 15
    emulator.fit(X, y)
    for complexity, loss in zip(emulator.complexity_list, emulator.loss_list):
        if complexity in complexity_to_loss:
            # Assert that the new loss (with one more iteration) is equal or smaller than the previous loss
            assert round(loss, 2) <= round(complexity_to_loss[complexity], 2)
            # print(round(loss, 2), round(complexity_to_loss[complexity], 2))
    assert emulator.loss_list

def test_is_gaussian():
    X, y = load_X_and_y_for_test()
    X_variable_names = ['x']
    y_variable_name = 'y'
    emulator = Emulator(niterations=1, gaussian_fit=True,
                        X_variable_names_for_gaussian_fit=X_variable_names, y_variable_name_for_gaussian_fit=y_variable_name)
    emulator.fit(X, y, variable_names=X_variable_names)
    loss_train_computed = round(float(emulator.compute_selected_loss(X, y)), 2)
    loss_train = round(float(emulator.selected_loss_train), 2)
    assert loss_train_computed == loss_train
    emulator.predict(X)
    emulator.predict(X, index=0)


def test_loss():
    X, y = load_X_and_y_for_test()
    emulator = Emulator(niterations=1)
    emulator.fit(X, y)
    assert round(emulator.compute_selected_loss(X, y), 2) == round(emulator.selected_loss_train, 2)

def test_corner_case_tournament_selection_n():
    X, y = load_X_and_y_for_test()
    emulator = Emulator(niterations=1, tournament_selection_n=15, population_size=14)
    emulator.fit(X, y)

