import pytest

from data.utils_dataset.npp_season_v1 import get_dataset
from optimization.optmization_pipeline.optimization_pipeline_zoo_500 import optimization_types_500
from optimization.utils_params.utils_param_name_to_values import ParamNameToValues


@pytest.mark.parametrize("optimization_type", optimization_types_500)
def test_budget(optimization_type: type):
    dataset = get_dataset()
    optimization = optimization_type('best', ParamNameToValues.DEFAULT_CENTRED)
    assert optimization.get_budget(dataset.X_train, dataset.y_train, dataset.validation_mask,
             dataset.X_variable_names, dataset.X_units, dataset.y_units) == 500

