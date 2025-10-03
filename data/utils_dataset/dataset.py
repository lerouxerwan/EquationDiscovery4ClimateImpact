from dataclasses import dataclass
from typing import Optional

from data.utils_dataset.utils_dataset_values import load_dataset_values
from data.utils_dataset.validation_split import ValidationSplit
from data.utils_run.utils_key import get_hash_str
from utils.utils_log import log_info


@dataclass
class Dataset(object):
    csv_filename: str
    rcp_name_train: str
    rcp_name_test: Optional[str] = None,
    validation_size: float = 0.3
    validation_split: ValidationSplit = ValidationSplit.RANDOM

    def __post_init__(self):
        (self.X_train, self.y_train, self.X_test, self.y_test, self.years_train, self.years_test,
         self.X_units, self.y_units, self.X_labels, self.y_labels, self.X_variable_names, self.y_variable_names,
         self.validation_mask, self.nb_historical_years) = load_dataset_values(self.csv_filename, self.rcp_name_train, self.rcp_name_test, self.validation_size, self.validation_split)
        # Check and display
        self.check()
        log_info(str(self))

    def check(self):
        """Check values are consistent between themselves"""
        assert len(self.X_units) == self.X_train.shape[1] == self.X_test.shape[1]
        assert len(self.X_units) == len(self.X_labels) == len(self.X_variable_names)

    def __str__(self):
        return (f"Dataset to predict {self.y_variable_names[0]} with {self.X_train.shape[1]} features, "
                f"{self.X_train.shape[0]} train datapoints, {self.X_test.shape[0]} test datapoints, "
                f"{str(self.validation_split)} validation split")

    @property
    def target_label(self) -> str:
        return self.y_labels[0]

    @property
    def hash(self) -> str:
        return get_hash_str(self.X_train, self.y_train, self.validation_mask, self.X_test, self.y_test)
