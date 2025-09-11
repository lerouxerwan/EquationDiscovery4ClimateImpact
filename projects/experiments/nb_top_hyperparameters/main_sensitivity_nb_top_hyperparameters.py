from matplotlib import pyplot as plt

from data.utils_dataset.validation_split import ValidationSplit
from projects.paper.section_results.validation_workflow import ValidationWorkflow


def main_sensitivity_nb_top_hyperparameters():
    model_selection = ['best', 'validated'][0]
    validation_split = [ValidationSplit.START, ValidationSplit.SYMMETRICAL, ValidationSplit.END,
                         ValidationSplit.RCP_START, ValidationSplit.EXTREME][0]
    n_iter = 100
    nb_top_hyperparameters_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # Load a single marginal search, with the largest nb_top_hyperparameters
    validation_workflow = ValidationWorkflow(validation_split, n_iter, max(nb_top_hyperparameters_list), model_selection)
    sorted_param_names = validation_workflow.sorted_param_names
    top_emulator_with_search_marginal = validation_workflow.top_emulator_with_search_marginal
    # Create the plot
    rmse_test_list = []
    for nb_top_hyperparameters in nb_top_hyperparameters_list:
        validation_workflow = ValidationWorkflow(validation_split, n_iter, nb_top_hyperparameters, model_selection,
                                                 sorted_param_names=sorted_param_names,
                                                 top_emulator_with_search_marginal=top_emulator_with_search_marginal)
        rmse_test_list.append(validation_workflow.rmse_test)
    ax = plt.gca()
    ax.plot(nb_top_hyperparameters_list, rmse_test_list)
    ax.set_xlabel('Number of top hyperparameters')
    ax.set_xticks(nb_top_hyperparameters_list)
    ax.set_ylabel(f'RMSE test ({validation_workflow.dataset.y_units[0]})')
    plt.show()

if __name__ == '__main__':
    main_sensitivity_nb_top_hyperparameters()