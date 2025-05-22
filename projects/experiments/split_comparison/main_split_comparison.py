from collections import OrderedDict
import os.path as op

import numpy as np
import pandas as pd

from data.utils_dataset.dataset import Dataset
from data.utils_dataset.validation_split import ValidationSplit, validation_split_to_validation_name
from projects.experiments.split_comparison.utils_feature_dataset import get_feature_datasets
from projects.experiments.split_comparison.utils_split_comparison import get_params_search, \
    get_params_emulator, get_res
from utils.utils_latex import plot_df_latex, print_df_latex
from utils.utils_log import log_info


def main_split_comparison(show: bool = False, fast: bool = False):
    # Select and check validation split
    niter = 100
    validation_splits = [ValidationSplit.START, ValidationSplit.SYMMETRICAL,
                         ValidationSplit.END, ValidationSplit.RCP_START,
                         ValidationSplit.EXTREME][:]
    # Start loop
    all_series_rmse = []
    all_series_equations = []
    for validation_split in validation_splits:
        df_true, df_predict, df_infos = compute_dataframes(validation_split, niter)
        df_squared_error = (df_true - df_predict)**2
        series_rmse = df_squared_error.mean(axis=0).apply(np.sqrt)
        validation_name = validation_split_to_validation_name[validation_split]
        series_rmse.name = validation_name
        all_series_rmse.append(series_rmse)
        series_equation = df_infos.iloc[0]
        series_equation.name = validation_name
        all_series_equations.append(series_equation)
    # Compute df_equation
    df_equation = pd.concat(all_series_equations, axis=1)
    # Compute df_rmse
    df_rmse = pd.concat(all_series_rmse, axis=1)
    # Remove the line where the predict is perfect
    indexes_to_remove = [1, 6]
    ind = [True] * len(df_equation)
    for i in indexes_to_remove:
        log_info(f'Field to remove for the ranking: {df_rmse.index.values[i]}', )
        log_info(f'Equation found: {df_equation.iloc[i, 0]}')
        ind[i] = False
    ind = pd.Series(index=df_equation.index, data=ind)
    df_rmse = df_rmse.loc[ind]
    # Compute df_ranks
    df_ranks = df_rmse.rank(axis=1)
    df_ranks.loc['Mean rank'] = df_ranks.mean()
    # Rounds dataframes
    df_rmse = df_rmse.round(decimals=2)
    df_ranks = df_ranks.round(decimals=1)
    # Print all dataframes
    for df in [df_rmse, df_ranks]:
        df = df.astype(str).replace(r'\.0$', '', regex=True)
        df.index.name = 'variable'
        df = df.reset_index()
        print_df_latex(df)


def compute_dataframes(validation_split, niter) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    validation_name = validation_split_to_validation_name[validation_split]
    log_info(f"Compute dataframes for validation_split {validation_name}")
    csv_filename = f'{niter}_{validation_name}.csv'
    true_csv_filename = 'csv/true_' + csv_filename
    predict_csv_filename = 'csv/predict_' + csv_filename
    infos_csv_filename = 'csv/infos_' + csv_filename

    if not op.exists(true_csv_filename):
        dataset = Dataset("NPP_season.csv", "RCP85", "RCP45", 0.3, validation_split)
        physical_variable_name_to_y_test_true = OrderedDict()
        physical_variable_name_to_y_test_predict = OrderedDict()
        physical_variable_name_to_infos = OrderedDict()
        for i, feature_dataset in enumerate(get_feature_datasets(dataset)):
            physical_variable_name = feature_dataset.y_variable_names[0]
            y_test_true, y_test_predict, infos = get_res(feature_dataset, get_params_emulator(), get_params_search(niter))
            physical_variable_name_to_y_test_predict[physical_variable_name] = y_test_predict
            physical_variable_name_to_infos[physical_variable_name] = infos
            physical_variable_name_to_y_test_true[physical_variable_name] = y_test_true
        # Save dataframes
        df_true = pd.DataFrame.from_dict(physical_variable_name_to_y_test_true)
        df_true.to_csv(true_csv_filename)
        df_predict = pd.DataFrame.from_dict(physical_variable_name_to_y_test_predict)
        df_predict.to_csv(predict_csv_filename)
        df_infos = pd.DataFrame.from_dict(physical_variable_name_to_infos)
        df_infos.to_csv(infos_csv_filename)
    else:
        filenames = [true_csv_filename, predict_csv_filename, infos_csv_filename]
        df_true, df_predict, df_infos = [pd.read_csv(filename, index_col=0) for filename in filenames]
    return df_true, df_predict, df_infos


if __name__ == '__main__':
    b = False
    main_split_comparison(b ,b)
    # main_split_comparaison(b,b)