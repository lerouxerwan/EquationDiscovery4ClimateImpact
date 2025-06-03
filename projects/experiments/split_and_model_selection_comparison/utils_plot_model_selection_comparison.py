from enum import StrEnum

from data.utils_dataset.validation_split import ValidationSplit
from projects.experiments.split_and_model_selection_comparison.utils_errors import compute_dataframe_errors
from utils.utils_latex import print_df_latex


class Strategy(StrEnum):
    PYSR = "'best' fitted on train + validation"
    # Only the two following strategy change with the validation split
    TRUE = "ground truth value on validation"
    BEST = "prediction with 'best' model selection fitted on train"
    CUSTOM = "prediction with 'custom' model selection fitted on train (and optimized on validation)"


def plot_compare_model_selection(validation_split):
    # Compute errors
    df_absolute_percentages_pysr, df_rmse_pysr = compute_dataframe_errors('best', ValidationSplit.NONE)
    df_absolute_percentages_best, df_rmse_best = compute_dataframe_errors('best', validation_split)
    df_absolute_percentages_custom, df_rmse_custom = compute_dataframe_errors('custom', validation_split)
    #  Compute df_absolute_percentages
    #  Compute df_ranks
    # df_ranks_rmse = df_rmse.rank(axis=1)
    # df_ranks_rmse.loc['Mean rank'] = df_ranks_rmse.mean()
    # #  Rounds dataframes
    # df_rmse = df_rmse.round(decimals=2)
    # df_absolute_percentages = df_absolute_percentages.round(decimals=2)
    # df_ranks_rmse = df_ranks_rmse.round(decimals=1)
    # #  Create a column with
    # df_absolute_percentages['Mean'] = df_absolute_percentages.mean(axis=1)
    # df_absolute_percentages = df_absolute_percentages.sort_values(by="Mean", axis=0)
    # df_ranks_rmse = df_ranks_rmse.loc[df_absolute_percentages.index]
    # Print all dataframes
    print(f"Dataframes for validation split = '{validation_split}'")
    print(df_rmse_pysr)
    print(df_rmse_best)
    print(df_rmse_custom)
    # for df in [df_ranks_rmse, df_absolute_percentages]:
    #     df = df.astype(str).replace(r'\.0$', '', regex=True)
    #     df.index.name = 'variable'
    #     df = df.reset_index()
    #     print_df_latex(df)
