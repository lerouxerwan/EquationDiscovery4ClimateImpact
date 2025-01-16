from typing import Any

from emulator.climate_impact_emulator import ClimateImpactEmulator
from emulator.utils_plots.plot_loss_vs_complexity import plot_loss_vs_complexity
from emulator.utils_plots.plot_scatter import plot_scatter
from emulator.utils_plots.plot_time_series import plot_time_series


def plot_full_diagnosis(emulator: ClimateImpactEmulator, split_name_to_X_and_y_and_years: dict[str, Any], show: bool):
    """Generate many plots to generator a full diagnosis for the emulator"""
    split_name_to_x_and_y = {split_name: (X, y) for split_name, (X, y, _) in split_name_to_X_and_y_and_years.items()}
    # Plot loss versus complexity
    plot_loss_vs_complexity(emulator, split_name_to_x_and_y, show)
    # Plot scatter
    plot_scatter(emulator, split_name_to_x_and_y, show)
    # Plot time series
    plot_time_series(emulator, split_name_to_X_and_y_and_years, show)
