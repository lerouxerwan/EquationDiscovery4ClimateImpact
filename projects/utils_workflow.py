from data.dataset.utils_dataset import load_dataset_dataframe
from emulator.climate_impact_emulator_with_search import ClimateImpactEmulatorWithSearch
from emulator.utils_hyperparameter_search.utils_validation import get_X_and_y
from emulator.utils_plots.plot_full_diagnosis import plot_full_diagnosis
from emulator.utils_plots.utils_plot_split_name import SPLIT_NAMES


def workflow(filename: str, nb_features: int, show: bool = False, **params_emulator) -> None:
    """Workflow that fit an emulator and generate diagnosis plots to assess the quality of this emulator"""
    # Load dataset
    (X_train, y_train, X_test, y_test, X_units, y_units, years_train, years_test, variable_names,
     target_label, index_start_validation) = load_dataset_dataframe(filename)
    # Fit emulator with search
    emulator = ClimateImpactEmulatorWithSearch(select_k_features=nb_features, **params_emulator)
    emulator.fit(X_train, y_train, variable_names=variable_names, X_units=X_units, y_units=y_units,
                 index_start_validation=index_start_validation)
    X_train_train, y_train_train = get_X_and_y(X_train, y_train, emulator.ind_validation_, validation_set=False)
    X_train_validation, y_train_validation = get_X_and_y(X_train, y_train, emulator.ind_validation_, validation_set=True)
    # Generate plots based on the 3 splits
    split_names = SPLIT_NAMES
    X_list = [X_train_train, X_train_validation, X_test]
    y_list = [y_train_train, y_train_validation, y_test]
    years_list = [years_train[~emulator.ind_validation_], years_train[emulator.ind_validation_], years_test]
    split_name_to_X_and_y_and_years = {split_name: (X, y, years)
                             for split_name, X, y, years in zip(split_names, X_list, y_list, years_list)}
    plot_full_diagnosis(emulator, split_name_to_X_and_y_and_years, target_label, show)

