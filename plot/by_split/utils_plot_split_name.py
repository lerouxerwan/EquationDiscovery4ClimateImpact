from data.utils_dataset.dataset import Dataset
from data.utils_dataset.validation_split import get_train_label, get_validation_label

SPLIT_NAMES = ['train', 'validation', 'test']

def get_label_split_name(split_name: str, dataset: Dataset) -> str:
    assert split_name in SPLIT_NAMES
    label = f'{split_name.capitalize()}, '
    if split_name == 'train':
        return label + get_train_label(get_rcp_name_with_dot(dataset.rcp_name_train),
                                       dataset.validation_split, dataset.validation_size)
    elif split_name == 'validation':
        return label + get_validation_label(get_rcp_name_with_dot(dataset.rcp_name_train),
                                            get_rcp_name_with_dot(dataset.rcp_name_test),
                                            dataset.validation_split, dataset.validation_size)
    else:
        return label + get_rcp_name_with_dot(dataset.rcp_name_test)
    
def get_rcp_name_with_dot(rcp_name: str) -> str:
    return f'{rcp_name[:-1]}.{rcp_name[-1]}'

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

split_name_to_hatch = {
    "train": None,
    "validation": "///",
      "test": ".",
}