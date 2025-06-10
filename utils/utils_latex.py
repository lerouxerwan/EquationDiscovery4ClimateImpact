import pandas as pd
from matplotlib import pyplot as plt

from utils.utils_plot import show_or_save_plot


def print_df_latex(df: pd.DataFrame):
    column_format = ''.join(['c' for _ in df.columns])
    s_latex = df.to_latex(index=False, column_format=column_format, float_format="%.2f")
    s = ') \\\\'
    s_latex = s_latex.replace(s, s + ' \\hline ')
    for s in ['\\midrule', '\\toprule']:
        s_latex = s_latex.replace(s, '\\hline \\hline ' + s)
    print('\n\n', s_latex, '\n\n')

def plot_df_latex(df_latex: pd.DataFrame, show: bool = False, fontsize=16):
    fix, ax = plt.subplots()
    ax.axis('off')
    table = pd.plotting.table(ax, df_latex, loc='center', cellLoc='center')
    table.auto_set_font_size(True)
    table.set_fontsize(fontsize)
    show_or_save_plot(f'selected_features', show)
