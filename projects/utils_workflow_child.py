from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

from emulator_with_search.climate_impact_emulator_with_search import ClimateImpactEmulatorWithSearch
from emulator.utils_plots.utils_plots import plot_diagnosis_fit
from emulator_with_search.search_dir.search_dir import SearchDir
from emulator_with_search.search_dir.utils_search_dir import METRIC_COLUMN_NAME
from utils.utils_dataset import load_dataset_dataframe


def workflow_child(filename: str, parent_search_dir: Optional[str] = None,
                   **params_emulator) -> None:
    """Workflow that fit an emulator and generate diagnosis plots to assess the quality of this emulator"""
    # Load dataset
    (X_train, y_train, X_test, y_test, X_units, y_units, years_train, years_test, rcp_name_train, rcp_name_test,
    variable_names, target_label, nb_historical_years) = load_dataset_dataframe(filename)
    # Combine params_emulator with best_params from parent_search_dir
    if parent_search_dir is not None:
        parent_best_params = SearchDir(parent_search_dir).best_params
        parent_best_params.update(params_emulator)
        params_emulator = parent_best_params
    # Fit emulator with search
    emulator = ClimateImpactEmulatorWithSearch(**params_emulator)
    emulator.fit(X_train, y_train, variable_names=variable_names, X_units=X_units, y_units=y_units,
                 index_start_validation=nb_historical_years)
    # Plots
    if parent_search_dir is not None:
        if len(emulator.param_grid) == 1:
            # Plot the variation for this hyperparameter
            param_name = list(emulator.param_grid)[0]
            df = emulator.search_dir_.df_cv_results_ranked
            params_list = df['params'].to_list()
            param_values = [params[param_name] for params in params_list]
            sorted_param_values = sorted(list(set(param_values)))
            min_loss_list = []
            for sorted_param_value in sorted_param_values:
                ind = pd.Series(index=df.index, data=[v == sorted_param_value for v in param_values])
                min_loss = -df.loc[ind, METRIC_COLUMN_NAME].max()
                min_loss_list.append(min_loss)
            ax = plt.gca()
            ax.plot(sorted_param_values, min_loss_list)
            plt.show()
    plot_diagnosis_fit(emulator, X_train, y_train, X_test, y_test, years_train, years_test, rcp_name_train, rcp_name_test, nb_historical_years, target_label, False)






