from data.utils_dataset.npp_season_v1 import get_dataset
from data.utils_dataset.validation_split import ValidationSplit
from optimization.utils_params.utils_param_name_to_values import ParamNameToValues
from plot.plot_diagnosis import plot_diagnosis
from projects.paper.utils_paper import get_opt

if __name__ == '__main__':

    n_jobs = -1
    param_name_to_values = ParamNameToValues.DEFAULT_CENTRED_WO_OPERATORS
    validation_split = ValidationSplit.QUANTILE_WITH_BINNING
    opt = get_opt()

    # Load the good dataset with the top4 features
    dataset = get_dataset(validation_size=0.2, validation_split=validation_split)
    variable_names = ['SSS_MAM', 'SST_MAM', 'SST_DJF', 'Shortwave_DJF']
    indexes_variable_names = [dataset.X_variable_names.index(variable_name) for variable_name in variable_names]
    X_variable_names = [dataset.X_variable_names[index] for index in indexes_variable_names]
    X_train = dataset.X_train[:, indexes_variable_names]
    X_test = dataset.X_test[:, indexes_variable_names]
    X_units = [dataset.X_units[index] for index in indexes_variable_names]
    dataset.X_variable_names =X_variable_names
    dataset.X_train = X_train
    dataset.X_test = X_test
    dataset.X_units = X_units
    top_emulator, _ = opt.run(X_train, dataset.y_train, dataset.validation_mask,
                              X_variable_names, X_units, dataset.y_units)
    plot_diagnosis(top_emulator, dataset)
    # rmse_test = get_loss(opt, X_train, dataset.y_train, dataset.validation_mask,
    #                      X_variable_names, X_units, dataset.y_units,
    #                      X_test, dataset.y_test)
    # log_info(f'RMSE test for top emulator = {rmse_test}')