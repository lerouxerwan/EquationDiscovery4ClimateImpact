
def string_to_list_int(list_int_as_string: str) -> list[int]:
    return [int(i) for i in list_int_as_string[1:-1].split(', ')]