from sympy import Expr

from emulator.is_interpretable import apply_func_on_sub_expressions, get_pow


def is_linear(expr: Expr):
    result = apply_func_on_sub_expressions(expr, _is_linear)
    return all(result) if isinstance(result, list) else result

def _is_linear(sub_expr: Expr):
    pow = get_pow(sub_expr)
    if pow is None:
        return True
    else:
        exponent = pow.args[1]
        return exponent == 1