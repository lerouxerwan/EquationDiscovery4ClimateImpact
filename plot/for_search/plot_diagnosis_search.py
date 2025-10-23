from typing import Optional

import pandas as pd

from data.utils_dataset.dataset import Dataset
from emulator.emulator import Emulator
from emulator.emulator_with_search import EmulatorWithSearch
from emulator.utils_hyperparameter_search.utils_column_names import COLUMN_NAMES
from plot.for_search.plot_search_1d import plot_diagnosis_search_1d
from plot.for_search.plot_selection_rate_features_top_equations import plot_selection_rate_features_top_equations
from utils.utils_latex import print_df_latex


def plot_diagnosis_search(emulator: Emulator, dataset: Dataset, show: Optional[bool] = False):
    assert isinstance(emulator, EmulatorWithSearch)
    plot_selection_rate_features_top_equations(emulator, dataset)
    plot_diagnosis_search_1d(emulator, dataset, show)
    plot_summary_run(emulator)

def plot_summary_run(emulator: EmulatorWithSearch) -> None:
    df_list = [emulator.run_.df_cv_results['params']]
    for model_selection in ['best', 'validated']:
        emulator.run_.model_selection = model_selection
        column_names = COLUMN_NAMES[:3]
        df_list.append(emulator.run_.df_cv_results.loc[:, column_names].copy())
    df = pd.concat(df_list, axis=1)
    df.rename(columns={c: c.replace('_', ' ') for c in df.columns }, inplace=True)
    df.rename(columns={c: c.replace('validation', 'val') for c in df.columns }, inplace=True)
    df.rename(columns={c: c.replace('complexity', 'C(f)') for c in df.columns }, inplace=True)
    print(df.head())
    print_df_latex(df)



