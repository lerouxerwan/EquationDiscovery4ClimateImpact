from utils.utils_json_loader import string_to_dict


def test_json_loader():
    dict_string = '{"a1": True, "a2": False, "a3": None}'
    d = string_to_dict(dict_string)
    assert d['a1'] is True
    assert d['a2'] is False
    assert d['a3'] is None
    dict_string = "{'a1': [True, False]}"
    d = string_to_dict(dict_string)
    assert len(d['a1']) == 2
    assert d['a1'][0] == 'True'
    assert d['a1'][1] == 'False'
    dict_string = "{'a1': None}"
    d = string_to_dict(dict_string)
    assert d['a1'] is None
    dict_string = "{'a1': np.float64(2354.2)}"
    d = string_to_dict(dict_string)
    assert d['a1'] == 2354.2
    dict_string = "{'a1': np.float64(6.877829196166807e-05)}"
    d = string_to_dict(dict_string)
    assert d['a1'] == 6.877829196166807e-05




