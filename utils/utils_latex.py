import pandas as pd
from matplotlib import pyplot as plt

from utils.utils_plot import show_or_save_plot


def print_df_latex(df: pd.DataFrame, column_format=None, index: bool =False):
    print(str_df_latex(column_format, df, index))


def str_df_latex(df: pd.DataFrame, column_format=None, index: bool =False, number_decimal: int = 2) -> str:
    if column_format is None:
        column_format = ''.join(['c' for _ in df.columns])
    if index:
        column_format += 'c'
    s_latex = df.to_latex(index=index, column_format=column_format, float_format=f"%.{2}f")
    s = ') \\\\'
    s_latex = s_latex.replace(s, s + ' \\hline ')
    for s in ['\\midrule', '\\toprule']:
        s_latex = s_latex.replace(s, '\\hline \\hline ' + s)
    return s_latex


def plot_df_latex(df_latex: pd.DataFrame, show: bool = False, fontsize=16):
    fix, ax = plt.subplots()
    ax.axis('off')
    table = pd.plotting.table(ax, df_latex, loc='center', cellLoc='center')
    table.auto_set_font_size(True)
    table.set_fontsize(fontsize)
    show_or_save_plot(f'selected_features', show)
