import pandas as pd

from optimization.utils_params.utils_param_name_to_values import get_param_name_to_values, ParamNameToValues
from utils.utils_latex import str_df_latex


def main_show_hyperparameters_values():
    param_name_to_values = {param_name: [v for v in values if v is not None]
                            for param_name, values in get_param_name_to_values(ParamNameToValues.DEFAULT_CENTRED_WO_OPERATORS).items()}
    # param_name_to_s = {param_name: '['  + ','.join([f'{v:.2f}' if isinstance(v, float) else str(v)  for v in values]) + ']'
    #                    for param_name, values in param_name_to_values.items()}
    nb_decimals = 4
    param_name_to_s = {param_name.replace('_', ' '): ['[' + str(round(min(values), nb_decimals))
                                                      + ', ' + str(round(max(values), nb_decimals)) + ']']
                       for param_name, values in param_name_to_values.items()}
    df = pd.DataFrame(param_name_to_s).transpose()
    df.reset_index(inplace=True)
    df.columns = ['Hyperparameter', 'Range of values']

    partie1 = df.iloc[:15]
    partie2 = df.iloc[15:]

    # Split the dataframe in a smaller index
    partie1.reset_index(drop=True, inplace=True)
    partie2.reset_index(drop=True, inplace=True)

    # Concaténation horizontale
    df_final = pd.concat([partie1, pd.DataFrame.from_dict({" ": [None] * 15}), partie2], axis=1)

    print(len(df_final))
    s = str_df_latex(df_final)
    s = s.replace('NaN', '')
    print(s)

if __name__ == '__main__':
    main_show_hyperparameters_values()