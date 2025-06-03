import numpy as np
import pandas as pd

from projects.experiments.split_and_model_selection_comparison.utlis_run_predictions import compute_dataframes_y


def compute_dataframe_errors(model_selection, validation_splits):
    all_series_rmse = []
    all_series_absolute_percentage = []
    for validation_split in validation_splits:
        df_true, df_predict_best, df_predict_custom = compute_dataframes_y(validation_split)
        if model_selection == 'best':
            df_predict = df_predict_best
        elif model_selection == 'custom':
            df_predict = df_predict_custom
        else:
            raise NotImplementedError
        df_squared_error = (df_true - df_predict) ** 2
        series_rmse = df_squared_error.mean(axis=0).apply(np.sqrt)
        series_absolute_percentage = (100 * (df_predict - df_true) / df_true).apply(np.abs).mean(axis=0)
        validation_name = str(validation_split)
        series_rmse.name = validation_name
        series_absolute_percentage.name = validation_name
        all_series_rmse.append(series_rmse)
        all_series_absolute_percentage.append(series_absolute_percentage)
    #  Compute df_rmse
    df_rmse = pd.concat(all_series_rmse, axis=1)
    df_absolute_percentages = pd.concat(all_series_absolute_percentage, axis=1)
    return df_absolute_percentages, df_rmse
