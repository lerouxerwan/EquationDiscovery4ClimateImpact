import pytest

from data.dataset.utils_dataset import load_dataset_dataframe, load_dataset_ndarray


@pytest.mark.parametrize("load_dataset_function", [load_dataset_dataframe, load_dataset_ndarray])
def test_load_dataset(load_dataset_function):
    filename_dataset = r"v5_NPPz_annual_season_GOL4_allDepths_HIST_20_RCP85_94_RCP45_94_all_25_month_season.csv"
    X_train, y_train, X_test, y_test, X_units, y_units, years_train, years_test, rcp_name_train, rcp_name_test, _, _, index_start_validation = load_dataset_function(filename_dataset)
    assert len(X_train) == len(y_train) == len(years_train) == 114
    assert len(X_test) == len(y_test) == len(years_test) == 94
    assert index_start_validation == 20
    assert (years_train[0] == 1986) and (years_train[-1] == 2099)
    assert (years_test[0] == 2006) and (years_test[-1] == 2099)
    assert rcp_name_train == 'RCP85'
    assert rcp_name_test == 'RCP45'
