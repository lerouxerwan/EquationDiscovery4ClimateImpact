import sys

from data.utils_dataset.validation_split import ValidationSplit
from projects.paper.section_results.validation_workflow import ValidationWorkflow


def main():
    if len(sys.argv) > 1:
        indices = [int(sys.argv[i]) for i in range(1, 6)]
    else:
        indices = [0, 0, 1000, 5, 5]
    print(f'Run with indices={indices}')

    # Transform indices as arguments
    model_selection = ['best', 'validated'][indices[0]]
    validation_split = [ValidationSplit.START, ValidationSplit.SYMMETRICAL, ValidationSplit.END,
                        ValidationSplit.RCP_START, ValidationSplit.EXTREME][indices[1]]
    n_iter = indices[2]
    nb_top_hyperparameters = indices[3]
    max_depth = indices[4]
    validation_size = 0.3
    # validation_size = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5][indices[4] - 1]

    # Run validation workflow
    # non_default_dict = {
    #     'maxsize': 15,
    #     'niterations': 1000,
    # }
    non_default_dict = {
        'maxdepth': max_depth,
    }
    # non_default_dict = None
    validation_split = ValidationWorkflow(validation_split, n_iter, nb_top_hyperparameters, model_selection,
                                          validation_size, non_default_dict=non_default_dict)
    print('Run validation workflow:')
    print('Equation:', validation_split.emulator.selected_expr)
    print('Complexity:', validation_split.emulator.selected_complexity)
    print('RMSE test:', validation_split.rmse_test)

if __name__ == '__main__':
    main()