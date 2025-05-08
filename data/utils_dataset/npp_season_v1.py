from data.utils_dataset.dataset import Dataset
from data.utils_dataset.validation_split import ValidationSplit

dataset_values_npp_season_v1 = Dataset("NPP_season.csv", "RCP85", "RCP45", 0.3, ValidationSplit.RCP_START)