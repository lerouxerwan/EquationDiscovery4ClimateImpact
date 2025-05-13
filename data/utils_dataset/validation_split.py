from enum import Enum


class ValidationSplit(Enum):
    RANDOM = 0
    RCP_START = 1
    END = 2
    START = 3
    SYMMETRICAL = 4
    RANDOM_DECADE = 5
    MAX = 6
    MIN = 7
    EXTREME = 8

validation_split_to_name = {
    ValidationSplit.RANDOM: 'random',
    ValidationSplit.RCP_START: 'start RCP',
    ValidationSplit.END: 'end',
    ValidationSplit.START: 'start',
    ValidationSplit.SYMMETRICAL: 'symmetrical',
    ValidationSplit.RANDOM_DECADE: 'random_decade',
    ValidationSplit.MAX: 'max',
    ValidationSplit.MIN: 'min',
    ValidationSplit.EXTREME: 'extreme',
}


