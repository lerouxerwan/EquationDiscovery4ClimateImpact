from data.utils_dataset.npp_season_v1 import dataset_npp_season_v1
from projects.optimization.utils_workflow_child import workflow_child


def main_plot_results(show: bool):
    search_path = "/home/e23lerou/Documents/EquationDiscovery4ClimateImpact/data/search/82421e3d24e9c7423c3d8163d183862b/1d5d5580db592e245fc2b47770a0cefa"
    workflow_child(dataset_npp_season_v1, search_path, None, show)

if __name__ == '__main__':
    main_plot_results(show=False)