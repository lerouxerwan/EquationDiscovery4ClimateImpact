# General column names
PARAMS_EMULATOR_COLUMN_NAME = 'params_emulator'

# Column names for a specific model_selection
RMSE_TRAIN_COLUMN_NAME = 'RMSE_train'
RMSE_VALIDATION_COLUMN_NAME = 'RMSE_validation'
COMPLEXITY_COLUMN_NAME = 'complexity'
EXPR_COLUMN_NAME = 'expr'
VARIABLE_NAMES_COLUMN_NAME = 'variable_names'

COLUMN_NAMES = [RMSE_TRAIN_COLUMN_NAME, RMSE_VALIDATION_COLUMN_NAME, COMPLEXITY_COLUMN_NAME, EXPR_COLUMN_NAME,
                VARIABLE_NAMES_COLUMN_NAME]

def get_cv_results_column_name(model_selection: str, column_name: str) -> str:
    assert column_name in COLUMN_NAMES
    return f'{model_selection}_{column_name}'

def get_cv_results_column_names(model_selection: str) -> list[str]:
    return [get_cv_results_column_name(model_selection, column_name) for column_name in COLUMN_NAMES]






