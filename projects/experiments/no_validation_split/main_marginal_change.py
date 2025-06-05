from plot.workflow import workflow
from projects.experiments.no_validation_split.utils_no_validation_split import dataset_npp_season_no_validation_split


def main_run_marginal_change_maxsize():
    workflow(dataset_npp_season_no_validation_split, {'maxsize':20})

def main_run_marginal_change_adaptive_parsimony_scaling():
    workflow(dataset_npp_season_no_validation_split, {'adaptive_parsimony_scaling':2080})

def main_run_marginal_change_warmup_maxsize_by():
    workflow(dataset_npp_season_no_validation_split, {'warmup_maxsize_by':0.5})

def main_run_marginal_change_niterations():
    workflow(dataset_npp_season_no_validation_split, {'niterations':10})

def main_run_marginal_change_sqrt():
    workflow(dataset_npp_season_no_validation_split, {'unary_operators':['sqrt']})

def main_run_marginal_change_exp():
    workflow(dataset_npp_season_no_validation_split, {'unary_operators':['exp']})

def main_run_marginal_change_log():
    workflow(dataset_npp_season_no_validation_split, {'unary_operators':['log']})

if __name__ == '__main__':
    # main_run_marginal_change_warmup_maxsize_by()
    # main_run_marginal_change_niterations()
    # main_run_marginal_change_exp()
    main_run_marginal_change_log()