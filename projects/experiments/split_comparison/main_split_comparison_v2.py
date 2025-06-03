import os
import os.path as op
from collections import OrderedDict

import numpy as np
import pandas as pd

from data.utils_dataset.dataset import Dataset
from data.utils_dataset.validation_split import ValidationSplit
from emulator.emulator_validated_with_search import EmulatorValidatedWithSearch
from plot.utils_plot import plot_diagnosis
from projects.experiments.feature_dataset.utils_feature_dataset import get_feature_datasets, get_all_datasets
from projects.experiments.split_comparison.strategy import get_y_test
from projects.utils_params import get_params_search, get_params_emulator
from utils.utils_latex import print_df_latex
from utils.utils_log import log_info


def main_split_comparison(show: bool = False, fast: bool = False):
    # Select and check validation split
    i = 5
    validation_splits = [ValidationSplit.RCP_START, ValidationSplit.END,
                         ValidationSplit.START, ValidationSplit.SYMMETRICAL,
                         ValidationSplit.EXTREME, ValidationSplit.NONE][i:i+1]
    # Start loop
    all_series_rmse = []
    all_series_absolute_percentage = []
    for validation_split in validation_splits:
        df_true, df_predict_best, df_predict_custom = compute_dataframes(validation_split)
        df_squared_error = (df_true - df_predict_best)**2
        series_rmse = df_squared_error.mean(axis=0).apply(np.sqrt)
        series_absolute_percentage = (100 * (df_predict_best - df_true) / df_true).apply(np.abs).mean(axis=0)
        validation_name = str(validation_split)
        series_rmse.name = validation_name
        all_series_rmse.append(series_rmse)
        all_series_absolute_percentage.append(series_absolute_percentage)

    # Compute df_rmse
    df_rmse = pd.concat(all_series_rmse, axis=1)
    # Compute df_absolute_percentages
    df_absolute_percentages = pd.concat(all_series_absolute_percentage, axis=1)
    # Compute df_ranks
    df_ranks_rmse = df_rmse.rank(axis=1)
    df_ranks_rmse.loc['Mean rank'] = df_ranks_rmse.mean()
    # Rounds dataframes
    df_rmse = df_rmse.round(decimals=2)
    df_absolute_percentages = df_absolute_percentages.round(decimals=2)
    df_ranks_rmse = df_ranks_rmse.round(decimals=1)
    # Create a column with
    df_absolute_percentages['Mean'] = df_absolute_percentages.mean(axis=1)
    df_absolute_percentages = df_absolute_percentages.sort_values(by="Mean", axis=0)
    df_ranks_rmse = df_ranks_rmse.loc[df_absolute_percentages.index]
    # Print all dataframes
    for df in [df_ranks_rmse, df_absolute_percentages]:
        df = df.astype(str).replace(r'\.0$', '', regex=True)
        df.index.name = 'variable'
        df = df.reset_index()
        print_df_latex(df)


def compute_dataframes(validation_split) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    validation_name = str(validation_split)
    log_info(f"Compute dataframes for validation_split {validation_name}")
    folder = f'runs/{validation_name}'
    true_csv_filename = folder + '/true.csv'
    predict_custom_csv_filename =  folder + '/predict_custom.csv'
    predict_best_csv_filename =  folder + '/predict_best.csv'
    # infos_csv_filename =  folder + '/infos.csv'

    if not op.exists(true_csv_filename):
        dataset = Dataset("NPP_season.csv", "RCP85", "RCP45", 0.3, validation_split)
        physical_variable_name_to_y_test_true = OrderedDict()
        physical_variable_name_to_y_test_predict_best = OrderedDict()
        physical_variable_name_to_y_test_predict_custom = OrderedDict()
        # physical_variable_name_to_infos = OrderedDict()
        for i, loop_dataset in enumerate(get_all_datasets(dataset)):
            physical_variable_name = loop_dataset.y_variable_names[0]
            y_test_true, y_test_predict_custom, y_test_predict_best = get_y_test(loop_dataset, folder)
            physical_variable_name_to_y_test_true[physical_variable_name] = y_test_true
            physical_variable_name_to_y_test_predict_best[physical_variable_name] = y_test_predict_best
            physical_variable_name_to_y_test_predict_custom[physical_variable_name] = y_test_predict_custom
            # if i == 1:
            #     break
            # physical_variable_name_to_infos[physical_variable_name] = infos
        # Save dataframes
        df_true = pd.DataFrame.from_dict(physical_variable_name_to_y_test_true)
        df_true.to_csv(true_csv_filename)
        df_predict_best = pd.DataFrame.from_dict(physical_variable_name_to_y_test_predict_best)
        df_predict_best.to_csv(predict_best_csv_filename)
        df_predict_custom = pd.DataFrame.from_dict(physical_variable_name_to_y_test_predict_custom)
        df_predict_custom.to_csv(predict_custom_csv_filename)
    else:
        filenames = [true_csv_filename, predict_best_csv_filename, predict_custom_csv_filename]
        df_true, df_predict_best, df_predict_custom = [pd.read_csv(filename, index_col=0) for filename in filenames]
    return df_true, df_predict_best, df_predict_custom



if __name__ == '__main__':
    b = False
    main_split_comparison(b ,b)
    # main_split_comparaison(b,b)