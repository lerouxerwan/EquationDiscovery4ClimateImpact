from dataclasses import dataclass

import numpy as np
from sympy import Expr, expand

from emulator.utils_variable_names import get_variable_signed_names_for_term


@dataclass
class OrderedEquation(object):
    """Equation where terms are ordered by their relative importance for the data X"""
    expr: Expr
    X: np.ndarray
    variable_names: list[str]

    def __str__(self):
        return self.get_str(self.sub_equations_normalized)

    @staticmethod
    def get_str(sub_equations: list[str]):
        return ' + '.join(sub_equations).replace('+ -', '- ')

    @property
    def sub_equations(self) -> list[str]:
        return [str(term) for term in self.sorted_terms]

    @property
    def sub_equations_normalized(self) -> list[str]:
        return [f'{weight}*({term / weight})' for weight, term in self.sorted_weight_and_term]

    # Sorted weights

    @property
    def sorted_weights(self) -> list[float]:
        return [w for w, _ in self.sorted_weight_and_term]

    @property
    def sorted_terms(self) -> list[Expr]:
        return [term for _, term in self.sorted_weight_and_term]

    @property
    def sorted_weight_and_term(self) -> list[tuple[float, Expr]]:
        return list(sorted(zip(self.average_weights, self.terms), reverse=True))

    @property
    def variable_signed_name_to_weight(self) -> dict[str, float]:
        variable_signed_name_to_weight = dict()
        for weight, term in self.sorted_weight_and_term:
            variable_signed_names = get_variable_signed_names_for_term(term)
            if len(variable_signed_names) == 1:
                variable_signed_name_to_weight[variable_signed_names[0]] = weight
        return variable_signed_name_to_weight

    @property
    def average_weights(self) -> list[float]:
        return np.mean(self.matrix_of_weights, axis=1)

    @property
    def matrix_of_weights(self):
        matrix = []
        for x in self.X:
            subs_dict = dict(zip(self.variable_names, x))
            absolute_weights = np.array([abs(term.subs(subs_dict)) for term in self.terms])
            relative_weights = absolute_weights / sum(absolute_weights)
            matrix.append(relative_weights)
        return np.transpose(np.array(matrix))

    @property
    def terms(self) -> list[Expr]:
        return list(expand(self.expr).args)

