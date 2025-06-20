import pandas as pd

from data.utils_dataset.validation_split import ValidationSplit
from plot.utils_plot import plot_diagnosis
from projects.simple_paper_v2.validation_workflow import ValidationWorkflow
from utils.utils_latex import print_df_latex


def load_setting(fast):
    # Load setting
    validation_splits = [ValidationSplit.START, ValidationSplit.SYMMETRICAL, ValidationSplit.END]
    n_iter = 20
    nb_top_hyperparameters = 5
    nb_hyperparameters = None  # run marginal search for all hyperparameters
    if fast:
        validation_splits = validation_splits[:2]
        n_iter = 2
        nb_top_hyperparameters = 1
        nb_hyperparameters = 2
    return n_iter, nb_hyperparameters, nb_top_hyperparameters, validation_splits


def main_get_top_emulator(fast: bool):
    n_iter, nb_hyperparameters, nb_top_hyperparameters, validation_splits = load_setting(fast)
    # Load sorted validation workflows
    validation_workflows = [ValidationWorkflow(validation_split, n_iter, nb_top_hyperparameters, nb_hyperparameters) for validation_split in validation_splits]
    sorted_validation_workflow = sorted(validation_workflows, key=lambda vw: vw.rmse_test)
    # Create array with a summary of all validation workflows
    df = pd.concat([validation_workflow.series_summary for validation_workflow in validation_workflows], axis=1).transpose()
    df.index.name = 'Validation set'
    df.reset_index(inplace=True)
    print(df.head())
    print_df_latex(df)
    # Generate diagnosis for the validation workflow that minimizes rmse test
    validation_workflow = sorted_validation_workflow[0]
    plot_diagnosis(validation_workflow.emulator, validation_workflow.dataset, False)


if __name__ == '__main__':
    main_get_top_emulator(fast=True)