from data.dataset.utils_dataset import load_dataset


def test_load_dataset():
    filename_dataset = r"v4_NPPz_annual_season_GOL4_allDepths_HIST_20_RCP85_94_RCP45_94_all_25_month_season.csv"
    X_train, y_train, X_test, y_test, X_units, y_units, _, index_start_validation = load_dataset(filename_dataset)
    assert len(X_train) == len(y_train) == 114
    assert len(X_test) == len(y_test) == 94
    assert index_start_validation == 20