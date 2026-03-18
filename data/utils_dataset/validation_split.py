from enum import StrEnum

from data.load_goy_simu_dat import split_name_to_goy_label_name


class ValidationSplit(StrEnum):
    RANDOM = 'random'
    RCP_START = 'start RCP'
    END = 'end'
    START = 'start'
    MIDDLE = 'middle'
    RANDOM_DECADE = 'random_decade'
    MAX = 'max'
    MIN = 'min'
    EXTREME = 'extreme'
    NONE = 'no validation set'
    PRELIMINARY_TEST = 'preliminary test'
    QUANTILE_WITH_BINNING = 'quantile_with_binning'
    HIST = 'hist'

def get_train_label(rcp_name_train: str, validation_split: ValidationSplit, validation_size: float) -> str:
    percent = f'{int(100 * (1 - validation_size))}%'
    if validation_split in [ValidationSplit.START, ValidationSplit.MIDDLE, ValidationSplit.END]:
        match validation_split:
            case ValidationSplit.START:
                key_word = 'middle and end'
            case ValidationSplit.END:
                key_word = 'start and middle'
            case ValidationSplit.MIDDLE:
                key_word = 'start and end'
        return f'{percent} at the {key_word} of the historical period + {rcp_name_train}'
    elif validation_split is ValidationSplit.RCP_START:
        return f'historical period + the middle and end of {rcp_name_train}'
    elif validation_split is ValidationSplit.EXTREME:
        return f'{percent} that are intermediary values for the historical period + {rcp_name_train}'
    elif validation_split is ValidationSplit.RANDOM:
        return f'{percent} randomly in the historical period + {rcp_name_train}'
    elif validation_split in {ValidationSplit.NONE, ValidationSplit.PRELIMINARY_TEST}:
        return f'historical period + {rcp_name_train}'
    elif validation_split is ValidationSplit.QUANTILE_WITH_BINNING:
        return f'{percent} selected by quantile binning'
    elif validation_split is ValidationSplit.HIST:
        return split_name_to_goy_label_name['train']
    else:
        raise NotImplementedError

def get_validation_label(rcp_name_train: str, rcp_name_test: str, validation_split: ValidationSplit, validation_size: float) -> str:
    percent = f'{int(100 * validation_size)}%'
    if validation_split in [ValidationSplit.START, ValidationSplit.MIDDLE, ValidationSplit.END]:
        key_word = 'middle' if validation_split is ValidationSplit.MIDDLE else str(validation_split)
        return f'{percent} at the {key_word} of the historical period + {rcp_name_train}'
    elif validation_split is ValidationSplit.RCP_START:
        return f'{percent} at the start of {rcp_name_train}'
    elif validation_split is ValidationSplit.EXTREME:
        return f'{percent} that are low and high values for the historical period + {rcp_name_train}'
    elif validation_split is ValidationSplit.RANDOM:
        return f'{percent} randomly sampled in the historical period + {rcp_name_train}'
        # return f'{percent} randomly sampled'
    elif validation_split is ValidationSplit.NONE:
        raise ValueError('This function should not have been called')
    elif validation_split is ValidationSplit.PRELIMINARY_TEST:
        return rcp_name_test
    elif validation_split is ValidationSplit.QUANTILE_WITH_BINNING:
        return f'{percent} selected by quantile binning in the historical period + {rcp_name_train}'
    elif validation_split is ValidationSplit.HIST:
        return split_name_to_goy_label_name['validation']
    else:
        raise NotImplementedError



