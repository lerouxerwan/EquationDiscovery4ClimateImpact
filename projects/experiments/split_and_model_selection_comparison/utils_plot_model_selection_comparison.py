import pandas as pd

from data.utils_dataset.validation_split import ValidationSplit
from projects.experiments.split_and_model_selection_comparison.utils_errors import compute_dataframe_errors
from utils.utils_latex import print_df_latex


def plot_compare_model_selection(validation_split: ValidationSplit):
    """ Compare 3 different approaches:
        PYSR = "'best' fitted on train + validation"
        B = BEST = "prediction with 'best' model selection fitted on train"
        C = CUSTOM = "prediction with 'custom' model selection fitted on train (and optimized on validation)"""
    # Compute errors
    df_absolute_percentages_pysr, df_rmse_pysr = compute_dataframe_errors('best', [ValidationSplit.NONE])
    df_absolute_percentages_best, df_rmse_best = compute_dataframe_errors('best', [validation_split])
    df_absolute_percentages_custom, df_rmse_custom = compute_dataframe_errors('custom', [validation_split])
    df = pd.concat([df_rmse_pysr, df_rmse_best, df_rmse_custom, df_rmse_best], axis=1)
    # Compute ratio columns
    df.iloc[:, 3] /= df.iloc[:, 2]
    for j in [1, 2]:
        df.iloc[:, j] /= df.iloc[:, 0]
    # Combine dataframes
    df.index.name = str(validation_split)
    df.columns = ['Base', 'Best/Base', 'Custom/Base', 'Best/Custom']
    df['Best<Base'] = 100 * (df['Best/Base'] < 1)
    df['Custom<Base'] = 100 * (df['Custom/Base'] < 1)
    df['Custom<Best'] = 100 * (df['Best/Custom'] > 1)
    df = df.sort_values(by='Best/Base', axis=0)
    df.loc['Mean'] = df.mean()
    df = df.round(2)
    # Plot df
    df = df.astype(str).replace(r'\.0$', '', regex=True)
    df = df.reset_index()
    print_df_latex(df)
