import os.path as op
from typing import Optional

import numpy as np
import pandas as pd
from pysr.utils import ArrayLike

from utils.utils_path import DATASET_CSV_PATH


def load_dataset(filename_dataset: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
Optional[ArrayLike[str]], Optional[ArrayLike[str]], np.ndarray, np.ndarray, list[str], int]:
    df = pd.read_csv(op.join(DATASET_CSV_PATH, filename_dataset), index_col=0)
    # Extract the row 'UNIT' then remove it from df (if the row exists)
    if 'UNIT' in df.index:
        series_units = df.loc['UNIT']
        X_units = series_units.iloc[1:].to_list()
        y_units = series_units.iloc[:1].to_list()
        df = df.iloc[1:, :]
    else:
        X_units, y_units = None, None
    y = df.iloc[:, 0].values
    X = df.iloc[:, 1:].values
    ind_test = df.index.str.startswith('RCP45')
    X_test, y_test = X[ind_test, :], y[ind_test]
    X_train, y_train = X[~ind_test, :], y[~ind_test]
    variable_names = df.columns[1:].values
    variable_names = [variable_name.replace(' ', '_') for variable_name in variable_names]
    assert df.index.values[0].startswith('HIST')
    index_start_validation = int(df.index.str.startswith('HIST').sum())
    years = np.array([int(i.split('_')[-1]) for i in df.index.values])
    years_train = years[~ind_test]
    years_test = years[ind_test]
    return X_train, y_train, X_test, y_test, X_units, y_units, years_train, years_test, variable_names, index_start_validation

if __name__ == '__main__':
    filename_dataset = r"v4_NPPz_annual_season_GOL4_allDepths_HIST_20_RCP85_94_RCP45_94_all_25_month_season.csv"
    filename_dataset = r"units_v1.csv"
    res = load_dataset(filename_dataset)
    print(res[-1])