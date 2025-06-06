from plot.workflow import workflow
from projects.experiments.no_validation_split.utils_no_validation_split import dataset_npp_season_no_validation_split

niterations = 400

def main_run_marginal_change_warmup_maxsize_by_25():
    workflow(dataset_npp_season_no_validation_split, {'warmup_maxsize_by':0.25, 'niterations':niterations})

def main_run_marginal_change_warmup_maxsize_by_75():
    workflow(dataset_npp_season_no_validation_split, {'warmup_maxsize_by': 0.75, 'niterations': niterations})

def main_run_marginal_change_less_adaptative_parsimony_scaling():
    workflow(dataset_npp_season_no_validation_split, {'adaptive_parsimony_scaling':520, 'niterations':niterations})

def main_run_marginal_change_more_maxsize():
    workflow(dataset_npp_season_no_validation_split, {'maxsize': 40, 'niterations':niterations})

def main_run_marginal_change_warmup_maxsize_by_bonus():
    workflow(dataset_npp_season_no_validation_split, {'warmup_maxsize_by': 0.0625, 'niterations': niterations})


if __name__ == '__main__':
    # main_run_marginal_change_warmup_maxsize_by_25()
    # main_run_marginal_change_warmup_maxsize_by_75()
    # main_run_marginal_change_less_adaptative_parsimony_scaling()
    main_run_marginal_change_warmup_maxsize_by_bonus()
