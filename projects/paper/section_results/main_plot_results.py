from data.utils_dataset.npp_season_v1 import dataset_npp_season_v1
from projects.optimization.utils_workflow_child import workflow_child


def main_plot_results(show: bool):
    search_path = "/home/e23lerou/Documents/EquationDiscovery4ClimateImpact/data/search/ed87f1c21c867304d0c5b1fdd6062839/b1f603382535a297e302ebec8e7961d4"
    workflow_child(dataset_npp_season_v1, search_path, None, show)

if __name__ == '__main__':
    main_plot_results(show=False)