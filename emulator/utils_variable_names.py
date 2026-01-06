import operator
from functools import reduce
from itertools import chain

from sympy import Expr, Symbol, postorder_traversal, Integer, Float, Rational

from emulator.is_interpretable import is_interpretable, apply_func_on_sub_expressions


def get_variable_names(expr: Expr) -> list[str]:
    return list(set([str(s) for s in expr.atoms(Symbol)]))

def get_variable_signed_names(expr: Expr) -> list[str]:
    # So far this function is only implemented for the interpretable case
    if not is_interpretable(expr):
        raise NotImplementedError
    # Apply function '_get_variable_signed_names' on sub expressions
    variable_signed_names = apply_func_on_sub_expressions(expr, get_variable_signed_names_for_term)
    assert isinstance(variable_signed_names, list)
    if len(variable_signed_names) == 0:
        return []
    else:
        assert isinstance(variable_signed_names[0], (str, list))
        if isinstance(variable_signed_names[0], str):
            return variable_signed_names
        else:
            return list(chain.from_iterable(variable_signed_names))

def get_variable_signed_names_for_term(term: Expr) -> list[str]:
    variable_names = get_variable_names(term)
    if len(variable_names) == 0:
        return []
    else:
        assert len(variable_names) == 1
        variable_name = variable_names[0]
        decomposition = list(postorder_traversal(term))
        sign_list = [get_sign(node) for node in decomposition]
        sign = reduce(operator.mul, sign_list, 1)
        signed_variable_name = ('+' if sign > 0 else '-') + variable_name
        return [signed_variable_name]

def get_sign(node):
    if isinstance(node, (Integer, Float, Rational)):
        return +1 if float(node) > 0 else -1
    else:
        return 1