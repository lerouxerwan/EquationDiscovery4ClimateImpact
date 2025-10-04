import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes

from data.utils_dataset.dataset import Dataset
from optimization.optimization import Optimization
from optimization.utils_nested_cv.run_nested_cv import run_nested_cv
from utils.utils_plot import show_or_save_plot


def compare_nested_cv(dataset: Dataset, optimizations: list[Optimization]):
    opt_id_to_label = {optimization.opt_id: optimization.label for optimization in optimizations}
    opt_id_to_color = {optimization.opt_id: optimization.color() for optimization in optimizations}
    # Create dataframe
    d = {optimization.opt_id: run_nested_cv(dataset, optimization) for optimization in optimizations}
    df = pd.DataFrame.from_dict(d)
    s_sorted_mean =  df.sum().sort_values()
    for nb_box_plot in [5, 10]:
        df_loop = df[s_sorted_mean.index.values[:nb_box_plot]]
        ax = plt.gca()
        sns.boxplot(ax=ax, data=df_loop.melt(), x="variable", y="value", palette=opt_id_to_color, whis=1000)
        ax.set_xticklabels([opt_id_to_label[opt_id] for opt_id in df_loop.columns])
        ax.set_ylabel('RMSE')
        # ax.set_xlabel("Optimization strategy")
        plt.xticks(rotation=0)
        plt.grid(axis='y')
        ax_twin = ax.twiny()
        ax_twin.set_xlim(ax.get_xlim())
        ax_twin.set_xticks(ax.get_xticks())
        mean_values, std_values = df_loop.mean().values, df_loop.std().values
        cv_values = 100 * std_values / mean_values
        upper_xticklabels = [f'Mean = {round(mean, 2)}\n CV = {round(cv, 0)}%'
                             for mean, cv in zip(mean_values, cv_values)]
        ax_twin.set_xticklabels(upper_xticklabels)
        add_legend(ax_twin, optimizations)
        show_or_save_plot('compare_nested_cv', show=True)

def add_legend(ax: Axes, optimizations: list[Optimization]):
    legend_handles, legend_labels = [], []
    optimization_types = {type(optimization) for optimization in optimizations}
    for optimization_type in optimization_types:
        legend_labels.append(optimization_type.legend_label())
        color = optimization_type.color()
        legend_handles.append(plt.Line2D([0], [0], marker='s', linestyle='', color=color,
                            markerfacecolor=color, markersize=10, alpha=0.5))
    ax.legend(legend_handles, legend_labels, loc='upper left')