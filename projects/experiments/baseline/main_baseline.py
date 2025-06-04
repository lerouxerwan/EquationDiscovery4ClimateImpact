from data.utils_dataset.dataset import Dataset
from data.utils_dataset.validation_split import ValidationSplit
from plot.workflow import workflow


def main_run_baseline():
    dataset = Dataset("NPP_season.csv", "RCP85", "RCP45", 0.3, ValidationSplit.NONE)
    workflow(dataset, {})

if __name__ == '__main__':
    main_run_baseline()