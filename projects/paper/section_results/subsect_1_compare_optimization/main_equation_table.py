from typing import OrderedDict, Counter

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from data.utils_dataset.npp_season_v1 import get_dataset
from data.utils_dataset.validation_split import ValidationSplit
from emulator.utils_variable_names import get_variable_signed_names
from optimization.optimization_marginal.optimization_marginal import OptimizationMarginal
from optimization.optimization_random.optimization_random_zoo import OptimizationRandom_200, OptimizationRandom_4, \
    OptimizationRandom_500
from optimization.optmization_pipeline.optimization_pipeline_zoo_500 import OptimizationPipelineMarginalRandom, \
    OptimizationPipelineRandom
from optimization.utils_optimization import get_loss
from optimization.utils_params.utils_param_name_to_values import ParamNameToValues
from plot.by_split.utils_equation_str import str_to_new_str
from projects.paper.utils_paper import validation_splits
from utils.utils_date import get_season_short_names
from utils.utils_latex import print_df_latex, str_df_latex
from utils.utils_log import log_info
from utils.utils_plot import show_or_save_plot




def main_equation_table(show: bool = False):
    for opt_type in [OptimizationPipelineRandom][:]:
        plot_equation_table(opt_type)

def plot_equation_table(opt_type: type):
    validation_name_to_equation = OrderedDict()
    for validation_size in [0.2, 0.25, 0.3][:]:
        opt = opt_type('best', ParamNameToValues.DEFAULT_CENTRED_WO_OPERATORS, n_jobs=-1, timeout_in_seconds=60 * 60,
                       interpretable_mode=True)
        for validation_split in validation_splits:
            dataset = get_dataset(validation_size=validation_size, validation_split=validation_split)
            emulator = opt.get_top_emulator(dataset.X_train, dataset.y_train, dataset.validation_mask,
                                            dataset.X_variable_names, dataset.X_units, dataset.y_units)
            key = f'{int(validation_size * 100)}\% of {str(validation_split).replace('_', ' ')}'
            validation_name_to_equation[key] = [emulator.selected_equation]

    #  Third graph that correspond to an array
    df_equation = pd.DataFrame.from_dict(validation_name_to_equation).transpose()
    df_equation.reset_index(inplace=True)
    df_equation.rename(columns={0: 'Validated equation, i.e. Equation selected by the validation scheme', 'index': 'Validation set'}, inplace=True)

    equation_str = str_df_latex(df_equation, column_format='ll', index=False)
    equation_str = equation_str.replace('& $', '& {\\small $')
    equation_str = equation_str.replace('.0', '')
    equation_str = equation_str.replace('-04', '-4')
    equation_str = equation_str.replace('-1SS', '-SS')
    equation_str = equation_str.replace('$ \\', '$ } \\')
    # Add textrm everywhere
    for X_variable_name in dataset.X_variable_names:
        variable_name = X_variable_name.split('_')[0]
        if variable_name in str_to_new_str:
            variable_name = str_to_new_str[variable_name]
        old = f'{variable_name}_'
        new = '\\textrm{' + variable_name + '}_'
        equation_str = equation_str.replace(old, new)
    for season_short_name in get_season_short_names() + ['Annual']:
        old = '_{' + season_short_name + '}'
        new = '_{\\textrm{' + season_short_name + '}}'
        equation_str = equation_str.replace(old, new)

    print(equation_str)



    # print('\\end{center}')


if __name__ == '__main__':
    main_equation_table()


