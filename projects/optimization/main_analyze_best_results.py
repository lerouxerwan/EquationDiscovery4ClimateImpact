from data.utils_search.utils_best_score import get_best_search_experiments
from data.utils_search.utils_search_history import get_search_history
from projects.paper.utils_paper import filename_dataset_paper
from utils.utils_bash_call import bash_call
from data.utils_dataset.utils_dataset import load_dataset



def main_analyze_best_results(nb_top_experiments: int = 5):
    #  Load dataset
    (X_train, y_train, X_test, y_test, X_units, y_units, years_train, years_test, rcp_name_train, rcp_name_test,
     variable_names, target_label, validation_mask) = load_dataset(filename_dataset_paper)
    #  Get best search experiments
    search_experiments = get_best_search_experiments(X_train, y_train, validation_mask, nb_top_experiments)
    print('Ranking of the best search experiments:')
    for rank, search_experiment in list(enumerate(search_experiments, 1))[::-1]:
        print(f'Rank #{rank} {search_experiment}')
    # Focus on the best search experiment
    best_search_experiment = search_experiments[0]
    print('\nHistory to obtain the best search experiment:')
    for j, search_experiment in enumerate(get_search_history(best_search_experiment), 1):
        print(f'Step #{j} {search_experiment}')
    print('Json file for the best search experiment:', best_search_experiment.filepath_non_default_params)
    print('Best run is open in tensorboard')
    bash_call(f'tensorboard --logdir {best_search_experiment.log_dir}')



if __name__ == '__main__':
    main_analyze_best_results()