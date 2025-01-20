from typing import Any

import pandas as pd

from emulator.climate_impact_emulator import ClimateImpactEmulator
from emulator.utils_plots.plot_pareto_equations.plot_loss_vs_complexity import plot_loss_vs_complexity
from emulator.utils_plots.plot_selected_equation.plot_scatter import plot_scatter
from emulator.utils_plots.plot_selected_equation.plot_time_series import plot_time_series


def plot_full_diagnosis(emulator: ClimateImpactEmulator, split_name_to_X_and_y_and_years: dict[str, Any],
                        target_label:str, show: bool):
    """Generate many plots to generator a full diagnosis for the emulator"""
    # Cast all X and y as ndarray (instead of Dataframe and Series) if it is not already done
    if isinstance(list(split_name_to_X_and_y_and_years.values())[0], pd.DataFrame):
        split_name_to_X_and_y_and_years = {split_name: (X.values, y.values, *others)
                                           for split_name, (X, y, *others) in split_name_to_X_and_y_and_years.items()}
    # Create split_name_to_x_and_y
    split_name_to_x_and_y = {split_name: (X, y) for split_name, (X, y, _) in split_name_to_X_and_y_and_years.items()}
    # Plot loss versus complexity
    plot_loss_vs_complexity(emulator, split_name_to_x_and_y, target_label, show)
    # Plot scatter
    plot_scatter(emulator, split_name_to_x_and_y, target_label, show)
    # Plot time series
    plot_time_series(emulator, split_name_to_X_and_y_and_years, target_label, show)
