from plot.workflow import workflow
from projects.experiments.preliminary.utils_dataset_preliminary import get_dataset_preliminary_test


def main_search_preliminary_fast():
    dataset = get_dataset_preliminary_test()
    param_grid = {'maxsize': [35, 40]}
    workflow(dataset, {'niterations': 2},
             {'param_grid': param_grid, 'search_style': 'grid'})

def main_search_preliminary_test():
    dataset = get_dataset_preliminary_test()
    param_grid = {'maxsize': [20, 30]}
    workflow(dataset, {}, {'param_grid': param_grid, 'search_style': 'grid'})

if __name__ == '__main__':
    main_search_preliminary_fast()
