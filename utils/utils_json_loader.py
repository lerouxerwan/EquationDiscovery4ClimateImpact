import json
from typing import Any


class JsonLoader(object):

    @classmethod
    def load(cls, dict_string: str):
        converted_dict_string = cls._convert(dict_string)
        d = json.loads(converted_dict_string)
        map = {'True': True,
               'False': False,
               'None': None}
        for k, v in d.items():
            if isinstance(v, str):
                if v in map:
                    d[k] = map[v]
        return d

    @classmethod
    def _convert(cls, dict_string: str):
        # Convert to proper json format
        dict_string = dict_string.replace("'", '"').replace('u"', '"')
        for symbol in [True, False, None]:
            str_symbol = str(symbol)
            dict_string = dict_string.replace(str_symbol, f'"{str_symbol}"')
            # When the boolean is a parameter of an object, we revert the previous operation
            dict_string = dict_string.replace(f'="{str_symbol}"', f'={str_symbol}')
        # Only keep the float that is inside "np.float64(...)"
        dict_string = dict_string.replace('np.float64(', '')
        dict_string = dict_string.replace(')', '')
        return dict_string


def string_to_dict(dict_string: str) -> dict[str, Any]:
    return JsonLoader.load(dict_string)