from data.utils_dataset.dataset import Dataset
from data.utils_dataset.npp_season_v1 import get_dataset
from data.utils_dataset.validation_split import ValidationSplit
from optimization.optimization import Optimization
from optimization.optimization_baseline import OptimizationBaseline
from optimization.utils_nested_cv.compare_nested_cv import compare_nested_cv



def main_compare_nested_cv():
    opt_list: list[Optimization] = []
    model_selection = 'best'
    opt_list.append(OptimizationBaseline(model_selection))
    dataset = get_dataset()
    compare_nested_cv(dataset, opt_list)
