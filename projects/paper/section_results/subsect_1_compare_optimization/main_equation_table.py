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
from utils.utils_latex import print_df_latex, str_df_latex
from utils.utils_log import log_info
from utils.utils_plot import show_or_save_plot




def main_equation_table(show: bool = False):
    validation_splits = [ValidationSplit.QUANTILE_WITH_BINNING, ValidationSplit.MIDDLE, ValidationSplit.RANDOM][:]
    for opt_type in [OptimizationPipelineRandom][:]:
        plot_equation_table(opt_type, validation_splits)

def plot_equation_table(opt_type: type, validation_splits: list[ValidationSplit]):
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
    df_equation.rename(columns={0: 'Best equation', 'index': 'Validation split'}, inplace=True)
    # print('\\begin{center}')
    # print('\\scriptsize')
    equation_str = str_df_latex(df_equation, column_format='ll', index=False)
    equation_str = equation_str.replace('& $', '& {\\tiny $')
    equation_str = equation_str.replace('$ \\', '$ } \\')
    print(equation_str)

    # print('\\end{center}')


if __name__ == '__main__':
    main_equation_table()


