from enum import StrEnum


class ValidationSplit(StrEnum):
    RANDOM = 'random'
    RCP_START = 'start RCP'
    END = 'end'
    START = 'start'
    SYMMETRICAL = 'symmetrical'
    RANDOM_DECADE = 'random_decade'
    MAX = 'max'
    MIN = 'min'
    EXTREME = 'extreme'
    NONE = 'no validation set'

def get_train_label(rcp_name_train: str, validation_split: ValidationSplit) -> str:
    return f'historical period + {rcp_name_train}'

def get_validation_label(rcp_name_train: str, validation_split: ValidationSplit) -> str:
    return f'historical period + {rcp_name_train}'


