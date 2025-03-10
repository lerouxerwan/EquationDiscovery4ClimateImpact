from emulator_with_search.utils_search_experiment.utils_best_score import get_best_search_experiments
from emulator_with_search.utils_search_experiment.utils_search_history import get_search_history
from projects.paper.utils_paper import filename_dataset_paper
from utils.utils_dataset import load_dataset


def main_get_history_best_search_experiment():
    best_search_experiment = _get_best_search_experiments(nb_top_experiments=1)[0]
    for j, search_experiment in enumerate(get_search_history(best_search_experiment), 1):
        print(f'Step #{j} {search_experiment}')

def main_get_best_search_experiments(nb_top_experiments: int = 5):
    search_experiments = _get_best_search_experiments(nb_top_experiments)
    for rank, search_experiment in list(enumerate(search_experiments, 1))[::-1]:
        print(f'Rank #{rank} {search_experiment}')
    print('Json file for the best search experiment:', search_experiments[0].filepath_non_default_params)
    print('Add here a display of the tree/history of experiments to get there')


def _get_best_search_experiments(nb_top_experiments):
    #  Load dataset
    (X_train, y_train, X_test, y_test, X_units, y_units, years_train, years_test, rcp_name_train, rcp_name_test,
     variable_names, target_label, ind_validation) = load_dataset(filename_dataset_paper)
    #  Get best search experiments
    return get_best_search_experiments(X_train, y_train, ind_validation, nb_top_experiments)


if __name__ == '__main__':
    # main_get_best_search_experiments()
    main_get_history_best_search_experiment()