import pandas as pd
from pandas import DataFrame

from data.utils_dataset.npp_season_v1 import get_dataset
from data.utils_dataset.validation_split import ValidationSplit
from projects.paper.section_results.subsect_3_analyze_best_equation.emulator_linear import EmulatorLinear




def get_df_rcsm4(scenario: str) -> pd.DataFrame:
    dataset = get_dataset(validation_split=ValidationSplit.NONE)
    variable_names = ['SSS_MAM', 'SST_MAM', 'SST_DJF', 'Shortwave_DJF']
    variable_indexes = [dataset.X_variable_names.index(variable_name) for variable_name in variable_names]
    emulator = EmulatorLinear(variable_names=variable_names, features_indexes=variable_indexes)
    emulator.fit(dataset.X_train, dataset.y_train)

    if scenario == "RCP45":
        first_index = list(dataset.years_test).index(2015)
        X = dataset.X_test[first_index:]
        y_predict = emulator.predict(X)
        years = dataset.years_test[first_index:]
    elif scenario == "RCP85":
        first_index = list(dataset.years_train).index(2015)
        X = dataset.X_train[first_index:]
        y_predict = emulator.predict(X)
        years = dataset.years_train[first_index:]
    else:
        raise ValueError(f"Scenario {scenario} not supported")

    d = {'year': years, 'NPP': y_predict}
    for variable_name, variable_index in zip(variable_names, variable_indexes):
        d[variable_name] = X[:, variable_index]
    df = pd.DataFrame.from_dict(d)
    df.set_index('year', inplace=True)
    return df


if __name__ == '__main__':
    for scenario in ['RCP45', 'RCP85']:
        print(get_df_rcsm4(scenario))
