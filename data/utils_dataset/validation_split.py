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

def get_train_label(rcp_name_train: str, validation_split: ValidationSplit, validation_size: float) -> str:
    percent = f'{int(100 * validation_size)}%'
    if validation_split in [ValidationSplit.START, ValidationSplit.SYMMETRICAL, ValidationSplit.END]:
        match validation_split:
            case ValidationSplit.START:
                key_word = 'middle and end'
            case ValidationSplit.END:
                key_word = 'start and middle'
            case ValidationSplit.SYMMETRICAL:
                key_word = 'start and end'
        return f'{percent} at the {key_word} of the historical period + {rcp_name_train}'
    elif validation_split is ValidationSplit.RCP_START:
        return f'historical period + the middle and end of {rcp_name_train}'
    elif validation_split is ValidationSplit.EXTREME:
        return f'{percent} that are intermediary values for the historical period + {rcp_name_train}'
    elif validation_split is ValidationSplit.RANDOM:
        return f'{percent} randomly in the historical period + {rcp_name_train}'
    elif validation_split is ValidationSplit.NONE:
        return 'historical period + {rcp_name_train}'
    else:
        raise NotImplementedError

def get_validation_label(rcp_name_train: str, validation_split: ValidationSplit, validation_size: float) -> str:
    percent = f'{int(100 * validation_size)}%'
    if validation_split in [ValidationSplit.START, ValidationSplit.SYMMETRICAL, ValidationSplit.END]:
        key_word = 'middle' if validation_split is ValidationSplit.SYMMETRICAL else str(validation_split)
        return f'{percent} at the {key_word} of the historical period + {rcp_name_train}'
    elif validation_split is ValidationSplit.RCP_START:
        return f'{percent} at the start of {rcp_name_train}'
    elif validation_split is ValidationSplit.EXTREME:
        return f'{percent} that are low and high values for the historical period + {rcp_name_train}'
    elif validation_split is ValidationSplit.RANDOM:
        return f'{percent} randomly in the historical period + {rcp_name_train}'
    elif validation_split is ValidationSplit.NONE:
        raise ValueError('This function should not have been called')
    else:
        raise NotImplementedError



