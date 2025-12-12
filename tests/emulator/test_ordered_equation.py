import numpy as np
from sympy import symbols

from emulator.equation import OrderedEquation


def test_ordered_equation():
    y, z = symbols('y z')
    expr = 1 + 3 * y + 2 * z
    X = np.array([
        [2, 4],
        [2, 4],
        [2, 4],
    ])
    ordered_equation = OrderedEquation(expr, X, ['y', 'z'])
    assert len(ordered_equation.terms) == len(ordered_equation.average_weights)
    assert ordered_equation.get_str(ordered_equation.sub_equations) == '2*z + 3*y + 1'


def test_ordered_equation_v2():
    y, z = symbols('y z')
    expr = 1 + 3 * y - 2 * z
    X = np.array([
        [2, 4],
        [2, 4],
        [2, 4],
    ])
    ordered_equation = OrderedEquation(expr, X, ['y', 'z'])
    assert len(ordered_equation.terms) == len(ordered_equation.average_weights)
    assert ordered_equation.get_str(ordered_equation.sub_equations) == '-2*z + 3*y + 1'

def test_ordered_equation_v3():
    y, z = symbols('y z')
    expr = 1 - 3 * y - 2 * z
    X = np.array([
        [2, 4],
        [2, 4],
        [2, 4],
    ])
    ordered_equation = OrderedEquation(expr, X, ['y', 'z'])
    assert len(ordered_equation.terms) == len(ordered_equation.average_weights)
    assert ordered_equation.get_str(ordered_equation.sub_equations) == '-2*z - 3*y + 1'