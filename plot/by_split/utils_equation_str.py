import re
from collections import Counter
from typing import Any

import numpy as np
from sympy import Expr, Number

from utils.utils_date import get_short_month_names, get_season_short_names


def get_equation(expr: Expr) -> str:
    equation_str = str(round_expr(expr))
    # Replace the month or the season
    for short_name in get_short_month_names() + get_season_short_names():
        equation_str = equation_str.replace(f'_{short_name}', '_{' + short_name + '}')
    # Remove the "_" after "Max", "Min" and "Mean"
    for s in ["Max", "Min", "Mean"]:
        equation_str = equation_str.replace(f'{s}_', s)
    # Split the equation on 2 lines if it is too long
    # return text_on_two_lines_if_too_long(equation_str)
    return equation_str

def replace_julia_square_by_python_power(equation: str):
    """Replace 'square(0.5 * x - y) by (0.5*x-y)**2 everywhere in the string"""
    # Tant qu'il y a des "square(" dans le texte
    while 'square(' in equation:
        # On cherche la dernière parenthèse ouvrante "square("
        start = equation.rfind('square(')
        if start == -1:
            break
        # On trouve la parenthèse fermante correspondante
        pile = 1  # On commence à 1 car on a déjà trouvé un "square("
        end = start + len('square(')
        while pile > 0 and end < len(equation):
            if equation[end] == '(':
                pile += 1
            elif equation[end] == ')':
                pile -= 1
            end += 1
        if pile != 0:
            break  # Parentheses mal équilibrées
        # On extrait le contenu entre parenthèses
        contenu = equation[start + len('square('):end - 1]
        # On remplace ce "square(contenu)" par "(contenu)**2"
        equation = equation[:start] + f"({contenu})**2" + equation[end:]

    return equation



def get_bold_equation(equation_str: str) -> str:
    return '$\\mathbf{' + equation_str + '}$'


def round_expr(expr: Expr) -> Expr:
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
