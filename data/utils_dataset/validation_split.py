from enum import Enum


class ValidationSplit(Enum):
    RANDOM = 0
    RCP_START = 1
    END = 2

validation_split_to_name = {
    ValidationSplit.RANDOM: 'random',
    ValidationSplit.RCP_START: 'start RCP',
    ValidationSplit.END: 'end RCP',
}


