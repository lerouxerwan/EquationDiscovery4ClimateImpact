from typing import Optional

import numpy as np
import pandas as pd

from emulator.climate_impact_emulator import ClimateImpactEmulator
from emulator.utils_plots.plot_by_rcp.utils_rcp import rcp_name_to_color
from emulator.utils_plots.plot_by_split.utils_plot_by_split import set_default_years


def load_rcp_name_to_list_of_years_and_y_and_color(emulator: ClimateImpactEmulator, y_train: np.ndarray | pd.Series,
                                                   y_test: Optional[np.ndarray | pd.Series] = None,
                                                   years_train: Optional[np.ndarray]=None, years_test: Optional[np.ndarray]=None, rcp_name_train: str='RCP85',
                                                   rcp_name_test: Optional[str]=None, nb_historical_years: int = 0):
    rcp_name_to_list_of_years_and_y_and_color = dict()
    # Some checks
    assert y_train.ndim == 1
    assert (years_train is None) or (years_train.ndim == 1)
    # Cast all y as ndarray (instead of Series) if it is not already done
    if isinstance(y_train, pd.Series):
        y_train, y_test = y_train.values, y_test.values
    # Set default for years_train and years_test if needed
    years_test, years_train = set_default_years(y_test, y_train, years_test, years_train)
    # Add rcp_name_train
    rcp_name_to_list_of_years_and_y_and_color[rcp_name_train] = [
        (years_train[:nb_historical_years], y_train[:nb_historical_years], 'k'),
        (years_train[nb_historical_years:], y_train[nb_historical_years:], rcp_name_to_color[rcp_name_train])
    ]
    # Add rcp_name_test
    if rcp_name_test is not None:
        rcp_name_to_list_of_years_and_y_and_color[rcp_name_test] = [
            (years_train[:nb_historical_years], y_train[:nb_historical_years], 'k'),
            (years_test, y_test, rcp_name_to_color[rcp_name_test])
        ]
    # Some final checks
    for list_of_years_and_y_and_color in rcp_name_to_list_of_years_and_y_and_color.values():
        for years, y, _ in list_of_years_and_y_and_color:
            assert len(years) == len(y)
    return rcp_name_to_list_of_years_and_y_and_color
