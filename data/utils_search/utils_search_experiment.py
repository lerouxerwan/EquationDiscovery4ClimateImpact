from utils.utils_log import log_info


def string_to_list_int(list_int_as_string: str) -> list[int]:
    try:
        return [int(i) for i in list_int_as_string[1:-1].split(', ')]
    except ValueError as e:
        log_info(f'{e.__repr__()} for the string {list_int_as_string}')
        return []