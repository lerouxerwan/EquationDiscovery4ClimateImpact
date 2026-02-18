from itertools import product

import numpy as np
from sympy import symbols, parse_expr, Expr

from data.utils_dataset.npp_season_v1 import get_dataset
from data.utils_run.run import Run
from emulator.emulator import Emulator
from emulator.emulator_with_search import EmulatorWithSearch
from emulator.is_interpretable import is_interpretable
from emulator.is_linear import is_linear
from optimization.utils_params.utils_param_name_to_values import ParamNameToValues
from projects.paper.utils_paper import validation_sizes, validation_splits, opt_type


def main_check_equations(show: bool):
    datasets = [get_dataset(validation_size=validation_size, validation_split=validation_split)
        for validation_split, validation_size in product(validation_splits, validation_sizes)]
    optimization= opt_type('best', ParamNameToValues.DEFAULT_CENTRED_WO_OPERATORS, n_jobs=1, timeout_in_seconds=60 * 60, interpretable_mode=True)
    for dataset in datasets:
        emulator = optimization.get_top_emulator(dataset.X_train, dataset.y_train, dataset.validation_mask,
                                                 dataset.X_variable_names, dataset.X_units, dataset.y_units)
        assert isinstance(emulator, EmulatorWithSearch)
        count_linear = 0
        complexity_list_for_non_linear_equations = []
        complexity_list_for_linear_equations = []
        df = emulator.run_.df_cv_results
        for equation, variable_names, complexity in zip(df['expr'], df['variable_names'], df['complexity']):
            if (len(variable_names) == 1) and (variable_names[0] == ''):
                pass
            else:
                _ = symbols(' '.join(variable_names))
            s = equation.replace('{', '').replace('}', '').replace('$\n$', '').replace('$', '')
            s = s.replace('^', '**')
            expr = parse_expr(s)
            assert isinstance(expr, Expr)
            if is_interpretable(expr):
                if is_linear(expr):
                    count_linear += 1
                    complexity_list_for_linear_equations.append(complexity)
                else:
                    print(expr)
                    complexity_list_for_non_linear_equations.append(complexity)

        print('Ratio of linear equations', 100 * count_linear / len(df))
        print('Average complexity for non linear equations:', np.mean(complexity_list_for_non_linear_equations))
        print('Average complexity for linear equations:', np.mean(complexity_list_for_linear_equations))

        """une idée plus avancée serait de verifier qu'il y a bien au moins une equation 
        non linear pour chaque pareto front quand la complexité augmente"""

if __name__ == '__main__':
    main_check_equations(False)