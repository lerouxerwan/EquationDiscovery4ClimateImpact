from typing import Optional

import pandas as pd

from data.utils_dataset.dataset import Dataset
from data.utils_run.run import Run
from emulator.emulator import Emulator
from emulator.emulator_with_search import EmulatorWithSearch
from emulator.utils_hyperparameter_search.utils_column_names import COLUMN_NAMES
from plot.for_search.plot_search_1d import plot_diagnosis_search_1d
from utils.utils_latex import print_df_latex


def plot_diagnosis_search(emulator: Emulator, dataset: Dataset, show: Optional[bool] = False, plot_folder: Optional[str] = None):
    assert isinstance(emulator, EmulatorWithSearch)
    # for nb_top_equations in [5, 10, 20]:
    #     plot_selection_rate_features_top_equations(dataset, emulator.run_, nb_top_equations, show, plot_folder)
    plot_diagnosis_search_1d(emulator.run_, emulator.param_grid, dataset.target_label,
                             show, plot_folder)
    plot_summary_run(emulator.run_)

def plot_summary_run(run: Run, show: Optional[bool] = False, plot_folder: Optional[str] = None) -> None:
    df_list = [run.df_cv_results['params']]
    for model_selection in ['best', 'validated']:
        run.model_selection = model_selection
        column_names = COLUMN_NAMES[:3]
        df_list.append(run.df_cv_results.loc[:, column_names].copy())
    df = pd.concat(df_list, axis=1)
    df.rename(columns={c: c.replace('_', ' ') for c in df.columns }, inplace=True)
    df.rename(columns={c: c.replace('validation', 'val') for c in df.columns }, inplace=True)
    df.rename(columns={c: c.replace('complexity', 'C(f)') for c in df.columns }, inplace=True)
    print(df.head())
    print_df_latex(df)



