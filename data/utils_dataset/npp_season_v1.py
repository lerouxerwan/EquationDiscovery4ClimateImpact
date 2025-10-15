from data.utils_dataset.dataset import Dataset
from data.utils_dataset.validation_split import ValidationSplit



def get_dataset(csv_name: str='NPP_season_and_annual_season', validation_size: float=0.3, validation_split: ValidationSplit=ValidationSplit.RCP_START) -> Dataset:
    return Dataset(f"{csv_name}.csv", "RCP85", "RCP45", validation_size, validation_split)