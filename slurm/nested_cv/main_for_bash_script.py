import sys

from data.utils_dataset.dataset import Dataset
from data.utils_dataset.validation_split import ValidationSplit
from optimization.optimization_marginal_search import OptimizationMarginalSearch
from optimization.utils_nested_cv import run_nested_cv
from projects.paper.section_results.validation_workflow import ValidationWorkflow
from slurm.nested_cv.utils_param_names import param_names


def main():
    if len(sys.argv) > 1:
        indices = [int(sys.argv[i]) for i in range(1, 3)]
    else:
        indices = [0, 0]
    print(f'Run with indices={indices}')

    # Transform index as argument
    model_selection_list = ['best', 'validated']
    model_selection = model_selection_list[indices[0]]
    param_name = param_names[indices[1]]

    dataset = Dataset("NPP_season.csv", "RCP85", "RCP45", 0.2, ValidationSplit.QUANTILE_WITH_BINNING)
    opt = OptimizationMarginalSearch(model_selection, param_name)
    run_nested_cv(dataset, opt, True)


if __name__ == '__main__':
    main()