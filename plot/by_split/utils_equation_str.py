from collections import Counter

import numpy as np
from sympy import Expr, Number

from utils.utils_date import get_short_month_names, get_season_short_names


def get_equation_str(expr: Expr, add_bold=False, add_underline=False) -> str:
    equation_str = str(round_expr_v3(expr))
    if add_bold:
        equation_str  = '$\\mathbf{' + equation_str + '}$'
        # equation_str  = '$\\mathbf{' + equation_str + '}$ (selected equation)'
    # elif add_underline:
    #     equation_str = '$\\mathbf{' + equation_str + '}$ (selected equation with PySR)'
    else:
        equation_str = f'${equation_str}$'
    # Replace the month or the season
    for short_name in get_short_month_names() + get_season_short_names():
        equation_str = equation_str.replace(f'_{short_name}', '_{' + short_name + '}')
    # Remove the "_" after "Max", "Min" and "Mean"
    for s in ["Max", "Min", "Mean"]:
        equation_str = equation_str.replace(f'{s}_', s)
    equation_str = text_on_two_lines_if_too_long(equation_str)
    return equation_str

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


def round_expr_v2(expr: Expr, num_digits: int) -> Expr:
    numbers = expr.atoms(Number)
    for number in numbers:
        round_number = round(number, num_digits)
        new_expr = expr.subs(number, round_number)
        if len(new_expr.atoms(Number)) == len(numbers):
            expr = new_expr
    return expr


def round_expr_v3(expr: Expr) -> Expr:
    numbers = expr.atoms(Number)
    for number in numbers:
        for num_digits in range(5):
            round_number = round(number, num_digits)
            new_expr = expr.subs(number, round_number)
            if len(new_expr.atoms(Number)) == len(numbers):
                expr = new_expr
                break
    return expr


def text_on_two_lines_if_too_long(text: str) -> str:
    if len(text) <= 100:
        return text
    else:
        characters = ['+', '-', '*']
        if any([character in text for character in characters]):
            index_plus_and_minus = [i for i, character in enumerate(text) if character in characters]
            # Remove some plus and minus indexes that may be between parenthesis (we do not want to cut there)
            index_with_left_parenthesis = [i for i, character in enumerate(text) if character in ['(']]
            index_with_right_parenthesis = [i for i, character in enumerate(text) if character in [')']]
            if len(index_with_left_parenthesis) > 0:
                for index_left, index_right in zip(index_with_left_parenthesis, index_with_right_parenthesis):
                    index_plus_and_minus = [i for i in index_plus_and_minus if not (index_left < i < index_right)]
            # Remove also plus and minus signs that are too close to the edge of the equation
            min_authorized_distance_to_the_edge = 10
            index_plus_and_minus = [i for i in index_plus_and_minus
                                    if (i > min_authorized_distance_to_the_edge)
                                    and  (i < len(text) - min_authorized_distance_to_the_edge)]
            if len(index_plus_and_minus) == 0:
                return text
            else:
                # Find the index plus and minus that is closest to the center of the text
                middle_index = len(text) // 2
                distance_to_middle_index = [abs(i - middle_index) for i in index_plus_and_minus]
                index_minimize_distance = index_plus_and_minus[np.argmin(distance_to_middle_index)]
                first_part, second_part = text[:index_minimize_distance], text[index_minimize_distance:]
                separator = '$\n$'
                counter_first_part = Counter(first_part)
                if counter_first_part['{'] > counter_first_part['}']:
                    separator = '}' + separator + '\\mathbf{'
                return first_part + separator + second_part
        else:
            return text
