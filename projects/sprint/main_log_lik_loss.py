import numpy as np
from pysr import TemplateExpressionSpec, PySRRegressor

from data.utils_dataset.dataset import Dataset
from data.utils_dataset.npp_season_v1 import get_dataset
from data.utils_dataset.validation_split import ValidationSplit

custom_function = """
function eval_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
    prediction, flag = eval_tree_array(tree, dataset.X, options)
    if !flag
        return L(Inf)
    end
    println(size(dataset.y))
    println(size(prediction))
    return sum((prediction .- dataset.y) .^ 2) / dataset.n
end
"""

template = TemplateExpressionSpec(
    expressions=["f", "g"],
    variable_names=["x_0", "x_1", "y"],
    combine = """
        mu = f(x_0, x_1)
        sigma = g(x_0, x_1)
        
        log(sigma) + (y - mu)^2 / (2 * sigma^2)
    """
)

emulator = PySRRegressor(niterations=1, expression_spec=template, elementwise_loss="my_custom_loss(predicted, target) = predicted")
dataset = get_dataset()
# Limit the number of featurest to two
X_train = np.concat([dataset.X_train[:, :2],  np.expand_dims(dataset.y_train, axis=1)], axis=1)
X_units = dataset.X_units[:2] + dataset.y_units
variable_names = ['x_0', 'x_1', 'y']
# Double the number of targets
y_train = np.expand_dims(dataset.y_train, axis=1)
y_train = np.concat([y_train, y_train], axis=1)
print(y_train.shape)
y_units = dataset.y_units + dataset.y_units
print(y_units)
emulator.fit(X_train, np.zeros(len(X_train)), variable_names=variable_names)
# plot_climato(emulator, dataset, show=True)