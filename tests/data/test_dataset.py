from data.utils_dataset.utils_dataset_values import load_dataset_values
from data.utils_dataset.validation_split import ValidationSplit


def test_load_dataset():
    (X_train, y_train, X_test, y_test, years_train, years_test,
        X_units, y_units, X_labels, y_labels, X_variables_names, y_variable_names,
        validation_mask) = load_dataset_values("NPP_season.csv", "RCP85", "RCP45", 0.3, ValidationSplit.RCP_START)
    any_value = X_train[0, 0]
    assert isinstance(any_value, float), type(any_value)
    assert len(X_train) == len(y_train) == len(years_train) == 114
    assert len(X_test) == len(y_test) == len(years_test) == 94
    assert (X_train.shape[1] == 96) and  (X_test.shape[1] == 96)
    assert (len(y_train.shape) == 1) and  (len(y_test.shape) == 1)
    assert sum(validation_mask) == 35
    assert (years_train[0] == 1986) and (years_train[-1] == 2099)
    assert (years_test[0] == 2006) and (years_test[-1] == 2099)
    for units in [X_units, y_units]:
        assert all([isinstance(unit, str) and (unit != 'nan') for unit in units])
