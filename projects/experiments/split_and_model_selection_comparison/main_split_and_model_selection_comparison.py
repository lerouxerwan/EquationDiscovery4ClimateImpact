from data.utils_dataset.validation_split import ValidationSplit
from projects.experiments.split_and_model_selection_comparison.utils_plot_model_selection_comparison import \
    plot_compare_model_selection
from projects.experiments.split_and_model_selection_comparison.utlis_plot_split_comparison import \
    plot_table_compare_split
from projects.experiments.split_and_model_selection_comparison.utlis_run_predictions import compute_dataframes_y

def main_run_predictions(show: bool = False, fast: bool = False):
    # Select and check validation split
    i = 5
    validation_splits = [ValidationSplit.RCP_START, ValidationSplit.END,
                         ValidationSplit.START, ValidationSplit.SYMMETRICAL,
                         ValidationSplit.EXTREME, ValidationSplit.NONE][i:i+1]
    # Generate csv files
    for validation_split in validation_splits:
        _ = compute_dataframes_y(validation_split)

def main_plot_comparison(show: bool = False, fast: bool = False):
    # Select and check validation split
    validation_splits = [ValidationSplit.RCP_START, ValidationSplit.END,
                         ValidationSplit.START, ValidationSplit.SYMMETRICAL,
                         ValidationSplit.EXTREME, ValidationSplit.NONE][1:-1]
    # Plot to compare split (2 comparisons, one for each model selection)
    for model_selection in ['best', 'custom']:
        plot_table_compare_split(validation_splits, model_selection)
    # Plot to compare model selection (5 comparisons, one for each validation split)
    # for validation_split in validation_splits[:-1]:
    #     plot_compare_model_selection(validation_split)


if __name__ == '__main__':
    b = False
    main_plot_comparison()