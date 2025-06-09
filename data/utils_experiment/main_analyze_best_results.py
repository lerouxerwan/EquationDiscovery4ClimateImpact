import subprocess

from data.utils_dataset.npp_season_v1 import dataset_npp_season_v1
from data.utils_experiment.utils_best_experiment import get_top_experiments
from data.utils_experiment.utils_history import get_history
from utils.utils_bash_call import bash_call


def main_analyze_best_results(nb_top_experiments: int = 5):
    #  Load dataset
    (X_train, y_train, *_, validation_mask) = dataset_npp_season_v1.values
    #  Get best search experiments
    search_experiments = get_top_experiments(X_train, y_train, validation_mask, nb_top_experiments)
    print('Ranking of the best search experiments:')
    for rank, search_experiment in list(enumerate(search_experiments, 1))[::-1]:
        print(f'Rank #{rank} {search_experiment}\n')
    # Focus on the best search experiment
    best_search_experiment = search_experiments[0]
    print('\nHistory to obtain the best search experiment:')
    for j, search_experiment in enumerate(get_history(best_search_experiment), 1):
        print(f'Step #{j} {search_experiment}')
    print('Json file for the best search experiment:', best_search_experiment.filepath_non_default_params)
    print('Best run is open in tensorboard')
    try:
        bash_call(f'tensorboard --logdir {best_search_experiment.log_dir}')
    except subprocess.CalledProcessError as e:
        print(e.__repr__())



if __name__ == '__main__':
    main_analyze_best_results()