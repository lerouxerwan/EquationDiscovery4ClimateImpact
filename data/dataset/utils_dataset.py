import os.path as op
from typing import Optional

import numpy as np
import pandas as pd
from pysr.utils import ArrayLike

from utils.utils_path import DATASET_CSV_PATH


def load_dataset(filename_dataset: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
Optional[ArrayLike[str]], Optional[ArrayLike[str]], list[str], int]:
    """

    :param filename_dataset:
    :return:
    """
    df = pd.read_csv(op.join(DATASET_CSV_PATH, filename_dataset), index_col=0)
    y = df.iloc[:, 0].values
    X = df.iloc[:, 1:].values
    ind_test = df.index.str.startswith('RCP45')
    X_test, y_test = X[ind_test, :], y[ind_test]
    X_train, y_train = X[~ind_test, :], y[~ind_test]
    variable_names = df.columns[1:].values
    variable_names = [variable_name.replace(' ', '_') for variable_name in variable_names]
    assert df.index.values[0].startswith('HIST')
    index_start_validation = int(df.index.str.startswith('HIST').sum())
    X_units = None
    y_units = None
    return X_train, y_train, X_test, y_test, X_units, y_units, variable_names, index_start_validation

if __name__ == '__main__':
    res = load_dataset(r"v4_NPPz_annual_season_GOL4_allDepths_HIST_20_RCP85_94_RCP45_94_all_25_month_season.csv")
    print(res[-1])