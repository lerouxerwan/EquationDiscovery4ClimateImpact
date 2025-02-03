import pandas as pd


def print_df_latex(df: pd.DataFrame):
    column_format = ''.join(['c' for _ in df.columns])
    s_latex = df.to_latex(index=False, column_format=column_format)
    s = ') \\\\'
    s_latex = s_latex.replace(s, s + ' \\hline ')
    for s in ['\\midrule', '\\toprule']:
        s_latex = s_latex.replace(s, '\\hline \\hline ' + s)
    print('\n\n', s_latex, '\n\n')