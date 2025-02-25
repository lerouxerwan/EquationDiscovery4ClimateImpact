SPLIT_NAMES = ['train', 'validation', 'test']

def get_label_split_name(split_name: str, rcp_name_train: str, rcp_name_test: str) -> str:
    assert split_name in SPLIT_NAMES
    label = f'{split_name.capitalize()}, '
    if split_name == 'train':
        return label + f'HIST + end of {rcp_name_train[:-1]}.{rcp_name_train[-1]}'
    elif split_name == 'validation':
        return label + f'start of {rcp_name_train[:-1]}.{rcp_name_train[-1]}'
    else:
        return label + f'{rcp_name_test[:-1]}.{rcp_name_test[-1]}'

split_name_to_linestyle = {
    "train": "-",
    "test": "dashed",
    "validation": "dotted"
}

split_name_to_marker = {
    "train": "o",
    "test": "s",
    "validation": "x"
}

split_name_to_color = {
    "train": "r",
    "validation": "orange",
      "test": "tab:purple",
}