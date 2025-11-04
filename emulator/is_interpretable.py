from typing import Any, Optional

from sympy import symbols, Expr, expand, Symbol, Basic, Add, Pow, Function, exp, log, sqrt
from sympy.utilities.misc import func_name


def is_interpretable(expr: Expr) -> bool:
    expanded_expr = expand(expr)
    if 'Add' in [func_name(a) for a in expanded_expr.atoms(Basic)]:
        return all([_is_interpretable(sub_expr)] for sub_expr in expanded_expr.args)
    else:
        return _is_interpretable(expanded_expr)

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
        return -2 <= exponent <= 2

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

if __name__ == '__main__':
    x, y = symbols('x y')
    expr = 1 + 2 * x + 3 * y + 4 * (x ** 2) + 5 / y
    # print(is_interpretable(expr))
    expr = sqrt(y)
    print(is_interpretable(expr))