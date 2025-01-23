import numpy as np
from sympy import Expr, Number, count_ops




def get_equation_str(expr: Expr) -> str:
    equation_str = str(round_expr(expr, 2))
    return text_on_two_lines_if_too_long(equation_str)

def round_expr(expr: Expr, num_digits: int) -> Expr:
    number_replacement = 1
    while number_replacement > 0:
        numbers = expr.atoms(Number)
        number_replacement = 0
        for number in numbers:
            round_number = round(number, num_digits)
            if round_number != number:
                number_replacement += 1
                expr = expr.subs(number, round_number)
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
