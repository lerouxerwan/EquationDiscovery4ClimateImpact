from data.utils_dataset.dataset import Dataset
from data.utils_dataset.validation_split import ValidationSplit
from emulator.emulator_with_search import EmulatorWithSearch
from plot.workflow import fit
from projects.simple_paper.search_strategy import SearchStrategy, get_params_emulator, get_params_search


def main_split_comparison(fast: bool = False):
    n_iter = 1 if fast else 10
    validation_splits = [ValidationSplit.START, ValidationSplit.SYMMETRICAL, ValidationSplit.END]
    search_strategies = [SearchStrategy.ALL_FULL_RANGE]
    # Fit emulator search for every validation_splits and every search_strategies
    for validation_split in validation_splits:
        for search_strategy in search_strategies:
            dataset = Dataset("NPP_season.csv", "RCP85", "RCP45", 0.3, validation_split)
            params_emulator = get_params_emulator(search_strategy)
            params_search = get_params_search(search_strategy)
            params_search['n_iter'] = n_iter
            emulator = EmulatorWithSearch(**params_emulator, **params_search)
            fit(emulator, dataset, refit=False)

if __name__ == '__main__':
    main_split_comparison(fast=True)