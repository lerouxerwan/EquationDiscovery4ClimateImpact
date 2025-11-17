from typing import Any, Optional, Callable, TypeVar

from sympy import symbols, Expr, expand, Symbol, Basic, Add, Pow, Function, exp, log, sqrt
from sympy.utilities.misc import func_name

T = TypeVar('T')

def apply_func_on_sub_expressions(expr: Expr, func: Callable[[Expr], T]) -> T:
    expanded_expr = expand(expr)
    if 'Add' in [func_name(a) for a in expanded_expr.atoms(Basic)]:
        return [func(sub_expr) for sub_expr in expanded_expr.args]
    else:
        return func(expanded_expr)

def is_interpretable(expr: Expr) -> bool:
    result = apply_func_on_sub_expressions(expr, _is_interpretable)
    assert isinstance(result, (bool, list)), result
    return all(result) if isinstance(result, list) else result

def _is_interpretable(sub_expr: Expr):
    # Sub expression with more than one variable is deemed non-interpretable
    if len(sub_expr.atoms(Symbol)) > 1:
        return False
    # Sub expression with function are deemed non-interpretable
    if len(sub_expr.atoms(Function)) > 0:
        return False
    # Otherwise, extract power expression from the sub expression, power between -2 and 2 is deemed interpretable
    pow =  get_pow(sub_expr)
    if pow is None:
        return True
    else:
        exponent = pow.args[1]
        return bool(-2 <= exponent <= 2)

def get_pow(sub_expr: Expr) -> Optional[Expr]:
    f_name = func_name(sub_expr)
    if f_name == 'Pow':
        return sub_expr
    elif f_name == 'Mul':
        for arg in sub_expr.args:
            if func_name(arg) == 'Pow':
                return arg
    # By default, we return None
    return None