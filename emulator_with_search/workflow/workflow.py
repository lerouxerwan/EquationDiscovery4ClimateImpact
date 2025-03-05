from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from emulator.utils_plots.utils_plots import plot_diagnosis_fit
from emulator_with_search.climate_impact_emulator_with_search import ClimateImpactEmulatorWithSearch
from emulator_with_search.utils_plots.plot_diagnosis_search import plot_diagnosis_search
from utils.utils_dataset import load_dataset_dataframe


@dataclass
class Workflow(ABC):
    """Workflow that fit an emulator with search to a dataset and generate diagnosis plots to assess fit quality"""
    dataset_filename: str

    def run(self):
        # Load dataset
        (X_train, y_train, X_test, y_test, X_units, y_units, years_train, years_test, rcp_name_train, rcp_name_test,
        variable_names, target_label, nb_historical_years) = load_dataset_dataframe(self.dataset_filename)
        # Fit emulator with search
        emulator = ClimateImpactEmulatorWithSearch(**self.params_emulator)
        emulator.fit(X_train, y_train, variable_names=variable_names, X_units=X_units, y_units=y_units,
                     index_start_validation=nb_historical_years)
        # Generate diagnosis plot
        plot_diagnosis_fit(emulator, X_train, y_train, X_test, y_test, years_train, years_test, rcp_name_train,
                           rcp_name_test, nb_historical_years, target_label, False)
        plot_diagnosis_search(emulator)


    @abstractmethod
    @property
    def params_emulator(self) -> dict[str, Any]:
        pass




