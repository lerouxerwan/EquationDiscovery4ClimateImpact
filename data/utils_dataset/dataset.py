from dataclasses import dataclass
from typing import Optional

from data.utils_dataset.utils_dataset_values import load_dataset_values
from data.utils_dataset.validation_split import ValidationSplit


@dataclass
class Dataset(object):
    csv_filename: str
    rcp_name_train: str
    rcp_name_test: Optional[str] = None,
    validation_size: float = 0.3
    validation_split: ValidationSplit = ValidationSplit.RANDOM

    def __post_init__(self):
        self.values = load_dataset_values(self.csv_filename, self.rcp_name_train, self.rcp_name_test, self.validation_size, self.validation_split)
        (self.X_train, self.y_train, self.X_test, self.y_test, self.years_train, self.years_test,
         self.X_units, self.y_units, self.X_labels, self.y_labels, self.X_variables_names, self.y_variable_names,
         self.validation_mask) = self.values