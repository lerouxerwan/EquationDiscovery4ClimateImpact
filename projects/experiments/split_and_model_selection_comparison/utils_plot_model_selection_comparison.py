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
    df = pd.concat([df_rmse_pysr, df_rmse_best, df_rmse_custom, df_rmse_best,
                    df_absolute_percentages_pysr, df_absolute_percentages_best, df_absolute_percentages_custom, df_absolute_percentages_best], axis=1)
    # Compute ratio columns
    for i in [0, 4]:
        df.iloc[i+3] /= df.iloc[i+2]
        for j in [i+1, i+2]:
            df.iloc[:, j] /= df.iloc[:, i]
    # Combine dataframes
    df.index.name = str(validation_split)
    df.columns = ['RMSE', 'B/', 'C/', 'B/C', 'Percent', 'Bp/', 'Cp/', 'Bp/Cp']
    df['B/>1'] = 100 * (df['B/'] > 1)
    df['C/>1'] = 100 * (df['C/'] > 1)
    df['Bp/>1'] = 100 * (df['Bp/'] > 1)
    df['Cp/>1'] = 100 * (df['Cp/'] > 1)
    df['B/C>1'] = 100 * (df['B/C'] > 1)
    df['Bp/Cp>1'] = 100 * (df['Bp/Cp'] > 1)
    # Do not keep all the columns
    df = df.loc[:, ['RMSE', 'B/', 'B/>1', 'C/', 'C/>1', 'B/C', 'B/C>1']]
    df = df.sort_values(by='B/', axis=0)
    df.loc['Mean'] = df.mean()
    df = df.round(2)
    # Plot df
    df = df.astype(str).replace(r'\.0$', '', regex=True)
    df = df.reset_index()
    print_df_latex(df)
