from matplotlib import pyplot as plt

from data.utils_dataset.validation_split import ValidationSplit
from projects.paper.section_results.validation_workflow import ValidationWorkflow
from utils.utils_log import log_info


def main_sensitivity_validation_size():
    model_selection = ['best', 'validated'][0]
    validation_split = [ValidationSplit.START, ValidationSplit.SYMMETRICAL, ValidationSplit.END,
                         ValidationSplit.RCP_START, ValidationSplit.EXTREME][0]
    n_iter = 100
    nb_top_hyperparameters = 5
    validation_size_list = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5][1:3]
    log_info(f'validation_size_list = {validation_size_list}')
    # Create the plot
    rmse_test_list = []
    for validation_size in validation_size_list:
        validation_workflow = ValidationWorkflow(validation_split, n_iter, nb_top_hyperparameters, model_selection,
                                                 validation_size)
        rmse_test_list.append(validation_workflow.rmse_test)
    ax = plt.gca()
    ax.plot(validation_size_list, rmse_test_list)
    ax.set_xlabel('Number of top hyperparameters')
    ax.set_xticks(validation_size_list)
    ax.set_ylabel(f'RMSE test ({validation_workflow.dataset.y_units[0]})')
    plt.show()

if __name__ == '__main__':
    main_sensitivity_validation_size()