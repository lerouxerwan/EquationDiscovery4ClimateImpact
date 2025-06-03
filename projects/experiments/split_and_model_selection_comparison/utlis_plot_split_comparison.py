from data.utils_dataset.validation_split import ValidationSplit
from projects.experiments.split_and_model_selection_comparison.utils_errors import compute_dataframe_errors
from utils.utils_latex import print_df_latex

def plot_table_compare_split(validation_splits, model_selection):
    # Do not consider Validation Split NONE here
    if ValidationSplit.NONE in validation_splits:
        validation_splits.remove(ValidationSplit.NONE)
    # Compute errors
    df_absolute_percentages, df_rmse = compute_dataframe_errors(model_selection, validation_splits)
    #  Compute df_absolute_percentages
    #  Order the rows by difficulty (first the easy target, at the end the difficult target in terms of absolute %)
    df_absolute_percentages['Mean'] = df_absolute_percentages.mean(axis=1)
    df_absolute_percentages = df_absolute_percentages
    df_absolute_percentages = df_absolute_percentages.sort_values(by="Mean", axis=0)
    df_absolute_percentages = df_absolute_percentages.drop('Mean', axis=1)
    # Compute the rank for the RMSE using the same ordering of the index
    df_rmse = df_rmse.loc[df_absolute_percentages.index]
    df_ranks_rmse = df_rmse.rank(axis=1)
    # Compute 'Mean' row for all the dataframe
    df_ranks_rmse.loc['Mean'] = df_ranks_rmse.mean()
    df_rmse.loc['Mean'] = df_rmse.mean()
    df_absolute_percentages.loc['Mean'] = df_absolute_percentages.mean()
    # Combine the dataframes
    df_summary = df_ranks_rmse.round(decimals=1).astype(str).map(lambda s: s.split('.')[0] if int(float(s)) == float(s) else s)
    df_summary += ' (' + df_rmse.round(decimals=2).astype(str)
    df_summary += ',' + df_absolute_percentages.round(decimals=1).astype(str) + '\%)'
    # Print all dataframes
    for df in [df_summary][:1]:
        df = df.astype(str).replace(r'\.0$', '', regex=True)
        df.index.name = f"Model selection = '{model_selection}'"
        df = df.reset_index()
        print_df_latex(df)
