from projects.utils_workflow import workflow
from utils.utils_run import NB_CORES

FILENAME = r"v5_NPPz_annual_season_GOL4_allDepths_HIST_20_RCP85_94_RCP45_94_all_25_month_season.csv"


def main_workflow(fast: bool = False):
    if fast:
        nb_features_list = [3]
        n_iter, n_jobs  = 2, 1
        param_grid = {'populations': [10, 20]}
    else:
        nb_features_list = [1, 2, 5]
        n_iter, n_jobs  = 2, NB_CORES
        param_grid = {
            'populations': [10, 20]
        }
    # Run workflow for several number of features
    for nb_features in nb_features_list:
        workflow(FILENAME, nb_features, param_grid, n_jobs, n_iter)


if __name__ == '__main__':
    fast = True
    main_workflow(fast)


