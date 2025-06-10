from plot.workflow import workflow
from projects.experiments.preliminary.utils_dataset_preliminary import get_dataset_preliminary_test


def main_search_preliminary_fast():
    dataset = get_dataset_preliminary_test()
    param_grid = {'niterations_warmup_maxsize': [2, 8]}
    workflow(dataset, {'niterations': 10},
             {'param_grid': param_grid, 'search_style': 'grid'})

def main_search_preliminary_test():
    dataset = get_dataset_preliminary_test()
    param_grid = {'niterations_warmup_maxsize': [10, 50]}
    workflow(dataset, {}, {'param_grid': param_grid, 'search_style': 'grid'})

def main_search_preliminary_rank_marginal():
    dataset = get_dataset_preliminary_test()
    param_grid = {'niterations_warmup_maxsize': [0, 10, 25, 50], 'adaptive_parsimony_scaling': [520., 1040.]}
    param_search = {'param_grid': param_grid, 'search_style': 'grid'}
    workflow(dataset, {'niterations': 400}, param_search)


if __name__ == '__main__':
    main_search_preliminary_rank_marginal()
    # main_search_preliminary_test()
    # main_search_preliminary_fast()
