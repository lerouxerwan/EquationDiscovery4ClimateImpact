from dataclasses import dataclass

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sympy.core.cache import cached_property

from data.utils_dataset.dataset import Dataset
from optimization.optimization import Optimization
from optimization.utils_nested_cv.run_nested_cv import run_nested_cv
from utils.utils_plot import show_or_save_plot


@dataclass
class CompareNestedCV:
    dataset: Dataset
    optimizations: list[Optimization]

    def __post_init__(self):
        self.opt_id_to_label = {optimization.opt_id: optimization.label for optimization in self.optimizations}
        self.opt_id_to_color = {optimization.opt_id: optimization.color for optimization in self.optimizations}

    def plot(self):
        print(self.df)
        s_sorted_mean =  self.df.sum().sort_values()
        df = self.df[s_sorted_mean.index]
        ax = plt.gca()
        sns.boxplot(ax=ax, data=df.melt(), x="variable", y="value", palette=self.opt_id_to_color)
        ax.set_xticklabels([self.opt_id_to_label[opt_id] for opt_id in df.columns])
        ax.set_ylabel('RMSE')
        ax.set_xlabel("Setting")
        plt.xticks(rotation=0)
        plt.grid(axis='y')
        ax_twin = ax.twiny()
        ax_twin.set_xlim(ax.get_xlim())
        ax_twin.set_xticks(ax.get_xticks())
        min_values, mean_values, max_values = df.min().values, df.mean().values, df.max().values
        mean_values = [f'{round(me, 2)} ({round(mi, 2)}, {round(ma, 2)})'
                       for mi, me, ma in zip(min_values, mean_values, max_values)]
        ax_twin.set_xticklabels([str(mean_value) for mean_value in mean_values], rotation=90)

        show_or_save_plot('compare_nested_cv', show=True)

    @cached_property
    def df(self):
        d = {optimization.opt_id: run_nested_cv(self.dataset, optimization) for optimization in self.optimizations}
        df = pd.DataFrame.from_dict(d)
        return df