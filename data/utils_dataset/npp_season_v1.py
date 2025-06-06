from data.utils_dataset.dataset import Dataset
from data.utils_dataset.validation_split import ValidationSplit

# This should only be imported in "main_*.py" files and from "test*.py" files (as it takes time to load the dataset)
dataset_npp_season_v1 = Dataset("NPP_season.csv", "RCP85", "RCP45", 0.3, ValidationSplit.RCP_START)