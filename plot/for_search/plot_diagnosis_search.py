from typing import Optional

import pandas as pd

from data.utils_dataset.dataset import Dataset
from data.utils_experiment.experiment import Experiment
from emulator.emulator import Emulator
from emulator.emulator_with_search import EmulatorWithSearch
from emulator.utils_hyperparameter_search.utils_column_names import get_cv_results_column_names
from plot.for_search.plot_search_1d import plot_diagnosis_search_1d
from plot.for_search.plot_selection_rate_features_top_equations import plot_selection_rate_features_top_equations
from utils.utils_latex import plot_df_latex, print_df_latex


def plot_diagnosis_search(emulator: Emulator, dataset: Dataset, show: Optional[bool] = False, plot_folder: Optional[str] = None):
    assert isinstance(emulator, EmulatorWithSearch)
    # for nb_top_equations in [5, 10, 20]:
    #     plot_selection_rate_features_top_equations(dataset, emulator.experiment_, nb_top_equations, show, plot_folder)
    plot_diagnosis_search_1d(emulator.experiment_, emulator.param_grid, dataset.target_label,
                             show, plot_folder)
    plot_summary_experiment(emulator.experiment_)

def plot_summary_experiment(experiment: Experiment, show: Optional[bool] = False, plot_folder: Optional[str] = None) -> None:
    df_list = [experiment.df_cv_results['params']]
    for model_selection in ['best', 'validated']:
        experiment.model_selection = model_selection
        column_names = get_cv_results_column_names(model_selection)[:3]
        df_list.append(experiment.df_cv_results.loc[:, column_names])
    df = pd.concat(df_list, axis=1)
    df.rename(columns={c: c.replace('_', ' ') for c in df.columns }, inplace=True)
    df.rename(columns={c: c.replace('validation', 'val') for c in df.columns }, inplace=True)
    df.rename(columns={c: c.replace('complexity', 'C(f)') for c in df.columns }, inplace=True)
    print(df.head())
    print_df_latex(df)



