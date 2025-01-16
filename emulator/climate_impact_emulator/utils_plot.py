from typing import Any

from emulator.climate_impact_emulator.climate_impact_emulator import ClimateImpactEmulator
from emulator.climate_impact_emulator.utils_plot_loss_vs_complexity import plot_loss_vs_complexity


def generate_plots(emulator: ClimateImpactEmulator, split_name_to_X_and_y_and_years: dict[str, Any], show: bool):
    # Plot loss versus complexity
    split_name_to_x_and_y = {split_name: (X, y) for split_name, (X, y, _) in split_name_to_X_and_y_and_years.items()}
    plot_loss_vs_complexity(emulator, split_name_to_x_and_y, show)
