from plot.workflow import workflow
from projects.experiments.no_validation_split.utils_no_validation_split import dataset_npp_season_no_validation_split


def main_run_baseline():
    workflow(dataset_npp_season_no_validation_split, {})

if __name__ == '__main__':
    main_run_baseline()