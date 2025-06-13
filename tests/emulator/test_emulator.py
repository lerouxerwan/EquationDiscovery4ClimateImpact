import numpy as np
import pytest
from sympy import Symbol

from data.utils_dataset.utils_validation import get_X_and_y
from emulator.emulator import Emulator
from emulator.emulator_with_search import EmulatorWithSearch
from emulator.utils_potential_contributions.utils_data_augmentation import apply_data_augmentation
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
    # Ensure that the two loss_list are consistent
    for loss_from_dataframe, loss_computed in zip(emulator.loss_list, loss_list):
        np.testing.assert_almost_equal(float(loss_from_dataframe), loss_computed)

@pytest.mark.repeat(2)
def test_deterministic_and_compute_loss():
    emulator = Emulator(niterations=1)
    X, y = load_X_and_y_for_test()
    emulator.fit(X, y)
    # Assert that the fit of the emulator is deterministic
    np.testing.assert_almost_equal(float(sum(emulator.loss_list)), 35698077.642941765)


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

def test_composed_units():
    emulator = Emulator(niterations=1, dimensionless_constants_only=True)
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

@pytest.mark.parametrize("emulator_type", [Emulator, EmulatorWithSearch])
def test_niterations_warmup_maxsize(emulator_type: type):
    # Cases with initialization
    assert emulator_type(niterations_warmup_maxsize=20, niterations=100).warmup_maxsize_by == 0.2
    assert emulator_type(niterations_warmup_maxsize=50, niterations=100).warmup_maxsize_by == 0.5
    assert emulator_type(niterations_warmup_maxsize=50, niterations=50).warmup_maxsize_by == 1.0
    assert emulator_type(niterations_warmup_maxsize=50, niterations=200).warmup_maxsize_by == 0.25
    # Cases with set_params
    emulator = emulator_type()
    emulator.set_params(niterations_warmup_maxsize=20)
    assert emulator.warmup_maxsize_by == 0.2
    emulator.set_params(niterations_warmup_maxsize=50)
    assert emulator.warmup_maxsize_by == 0.5
    # Cases expected to fail
    with pytest.raises(AssertionError):
        emulator_type(niterations_warmup_maxsize=-5, niterations=100)
    with pytest.raises(AssertionError):
        emulator_type(niterations_warmup_maxsize=120, niterations=100)




