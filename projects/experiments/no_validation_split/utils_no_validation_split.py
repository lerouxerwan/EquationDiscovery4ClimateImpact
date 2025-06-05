from data.utils_dataset.dataset import Dataset
from data.utils_dataset.validation_split import ValidationSplit

dataset_npp_season_no_validation_split = Dataset("NPP_season.csv", "RCP85", "RCP45", 0.3, ValidationSplit.NONE)
