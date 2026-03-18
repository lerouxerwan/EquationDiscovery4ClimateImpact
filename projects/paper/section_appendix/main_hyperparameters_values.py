import pandas as pd

from optimization.utils_params.utils_param_name_to_values import get_param_name_to_values, ParamNameToValues


def main_show_hyperparameters_values():
    param_name_to_values = get_param_name_to_values(ParamNameToValues.DEFAULT_CENTRED_WO_OPERATORS)
    # param_name_to_s = {param_name: '['  + ','.join([f'{v:.2f}' if isinstance(v, float) else str(v)  for v in values]) + ']'
    #                    for param_name, values in param_name_to_values.items()}
    param_name_to_s = {param_name: ['['  + ','.join([str(v)  for v in values]) + ']']
                       for param_name, values in param_name_to_values.items()}
    print(param_name_to_s)
    df = pd.DataFrame(param_name_to_s).transpose()
    print(df.head())

if __name__ == '__main__':
    main_show_hyperparameters_values()