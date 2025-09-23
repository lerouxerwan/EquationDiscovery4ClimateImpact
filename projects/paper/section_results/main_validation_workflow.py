import pandas as pd

from data.utils_dataset.validation_split import ValidationSplit
from plot.plot_diagnosis import plot_diagnosis
from projects.paper.section_results.validation_workflow import ValidationWorkflow
from utils.utils_latex import print_df_latex
from utils.utils_log import log_info


def load_setting(fast):
    model_selection = ['best', 'validated'][0]
    # Run independently the validation workflow, so that if it crashes, it does not crash everything
    val_id = 0
    validation_splits = [ValidationSplit.START, ValidationSplit.SYMMETRICAL, ValidationSplit.END,
                         ValidationSplit.RCP_START, ValidationSplit.EXTREME][val_id:val_id+1]
    n_iter = 100
    nb_top_hyperparameters = 5
    if fast:
        validation_splits = validation_splits[:2]
        n_iter = 2
        nb_top_hyperparameters = 1
    log_info(f'Start validation workflow with: '
             f'validation_splits={validation_splits} n_iter={n_iter} nb_top_hyperparameters={nb_top_hyperparameters}, '
             f'model_selection={model_selection}')
    return validation_splits, n_iter, nb_top_hyperparameters, model_selection


def main_get_top_emulator(fast: bool):
    validation_splits, n_iter, nb_top_hyperparameters, model_selection = load_setting(fast)
    # Load sorted validation workflows
    validation_workflows = [ValidationWorkflow(validation_split, n_iter, nb_top_hyperparameters,
                                               model_selection, validation_size=0.2)
                            for validation_split in validation_splits]
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
    main_get_top_emulator(fast=False)