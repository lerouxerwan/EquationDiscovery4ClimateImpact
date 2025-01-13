import os.path as op
import pandas as pd

from utils.utils_path import DATA_PATH


def load_data():
    filename = r"dataset/NPP/v4_NPPz_annual_season_GOL4_allDepths_HIST_20_RCP85_94_RCP45_94_all_25_month_season.csv"
    df = pd.read_csv(op.join(DATA_PATH, filename), index_col=0)
    y = df.iloc[:, 0].values
    X = df.iloc[:, 1:].values
    ind_test = df.index.str.startswith('RCP45')
    X_test, y_test = X[ind_test, :], y[ind_test]
    X_train, y_train = X[~ind_test, :], y[~ind_test]
    variable_names = df.columns[1:].values
    variable_names = [variable_name.replace(' ', '_') for variable_name in variable_names]
    assert df.index.values[0].startswith('HIST')
    index_start_validation = df.index.str.startswith('HIST').sum()
    X_units = None
    y_units = None
    return X_train, y_train, X_test, y_test, X_units, y_units, variable_names, index_start_validation

if __name__ == '__main__':
    load_data()