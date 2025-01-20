import numpy as np
from sympy import Expr, Number, count_ops




def get_equation_str(expr: Expr) -> str:
    equation_str = str(round_expr(expr, 1))
    return text_on_two_lines_if_too_long(equation_str)

def round_expr(expr: Expr, num_digits: int) -> Expr:
    n_list = expr.atoms(Number)
    for n in n_list:
        new_expr_ready = False
        expr_digits = num_digits
        while not new_expr_ready:
            #  Rounding is ok if it does not delete a term from the equation (and thus change the number of ops)
            new_expr = expr.subs(n, round(n, expr_digits))
            if count_ops(new_expr) == count_ops(expr):
                expr = new_expr
                new_expr_ready = True
            else:
                expr_digits += 1
        new_expr = expr.subs(n, round(n, num_digits))
        if count_ops(new_expr) == count_ops(expr):
            expr = new_expr
    return expr

def text_on_two_lines_if_too_long(text: str) -> str:
    if len(text) <= 50:
        return text
    else:
        characters = ['+', '-']
        if any([character in text for character in characters]):
            index_plus_and_minus = [i for i, character in enumerate(text) if character in characters]
            # Remove some plus and minus indexes that may be between parenthesis (we do not want to cut there)
            index_with_left_parenthesis = [i for i, character in enumerate(text) if character in ['(']]
            index_with_right_parenthesis = [i for i, character in enumerate(text) if character in [')']]
            if len(index_with_left_parenthesis) > 0:
                for index_left, index_right in zip(index_with_left_parenthesis, index_with_right_parenthesis):
                    index_plus_and_minus = [i for i in index_plus_and_minus if not (index_left < i < index_right)]
            if len(index_plus_and_minus) == 0:
                return text
            else:
                assert len(index_plus_and_minus) > 0
                middle_index = len(text) // 2
                distance_to_middle_index = [abs(i - middle_index) for i in index_plus_and_minus]
                index_minimize_distance = index_plus_and_minus[np.argmin(distance_to_middle_index)]
                return text[:index_minimize_distance] + '$\n$' + text[index_minimize_distance:]
        else:
            return text
