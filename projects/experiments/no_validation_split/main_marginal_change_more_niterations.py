from plot.workflow import workflow
from projects.experiments.no_validation_split.utils_no_validation_split import dataset_npp_season_no_validation_split

niterations = 2000

def main_run_marginal_change_niterations():
    workflow(dataset_npp_season_no_validation_split, {'niterations':niterations})

def main_run_marginal_change_warmup_maxsize_by():
    workflow(dataset_npp_season_no_validation_split, {'warmup_maxsize_by':0.5, 'niterations':niterations})

def main_run_marginal_change_sqrt():
    workflow(dataset_npp_season_no_validation_split, {'unary_operators':['sqrt'], 'niterations':niterations})

def main_run_marginal_change_log():
    workflow(dataset_npp_season_no_validation_split, {'unary_operators':['log'], 'niterations':niterations})

if __name__ == '__main__':
    # main_run_marginal_change_niterations()
    # main_run_marginal_change_warmup_maxsize_by()
    # main_run_marginal_change_sqrt()
    main_run_marginal_change_log()
