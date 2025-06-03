
from matplotlib import pyplot as plt

from data.utils_dataset.npp_season_v1 import dataset_npp_season_v1
from plot.dataset.plot_dataset import plot_values_target
from projects.experiments.feature_dataset.utils_feature_dataset import get_feature_datasets


def main_plot_feature_dataset():
    old_dataset = dataset_npp_season_v1
    for i, new_dataset in enumerate(get_feature_datasets(old_dataset)):
        if i == 15:
            plot_values_target(plt.gca(), new_dataset)
            plt.show()
        # if i == 1:
        #     break

if __name__ == '__main__':
    main_plot_feature_dataset()