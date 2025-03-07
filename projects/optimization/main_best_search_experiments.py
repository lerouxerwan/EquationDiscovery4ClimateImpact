from emulator_with_search.search_experiment.utils_search_experiment_with_best_score import get_best_search_experiment, \
    get_best_search_experiments
from projects.paper.utils_paper import filename_dataset_paper
from utils.utils_dataset import load_dataset


def main_get_best_search_experiments(nb_top_experiments):
    # Load dataset
    (X_train, y_train, X_test, y_test, X_units, y_units, years_train, years_test, rcp_name_train, rcp_name_test,
     variable_names, target_label, ind_validation) = load_dataset(filename_dataset_paper)
    # Show best search experiments
    search_experiments = get_best_search_experiments(X_train, y_train, ind_validation, nb_top_experiments)
    for rank, search_experiment in list(enumerate(search_experiments, 1))[::-1]:
        line = (f'#{rank} RMSE={round(search_experiment.best_rmse_validation, 3)} '
                f'with {search_experiment.best_expr} for {search_experiment.best_params}')
        print(line)
    print('Json file for the best search experiment:', search_experiments[0].filepath_non_default_params)
    print('Add here a display of the tree/history of experiments to get there')


if __name__ == '__main__':
    main_get_best_search_experiments(nb_top_experiments=1)