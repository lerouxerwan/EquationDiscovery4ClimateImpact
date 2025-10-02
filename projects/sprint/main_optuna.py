import optuna

from data.utils_dataset.dataset import Dataset
from data.utils_dataset.validation_split import ValidationSplit
from emulator.emulator import Emulator


def objective(trial):
    emulator = Emulator(
        niterations=trial.suggest_int("niterations", 10, 20),
        maxsize=trial.suggest_int("maxsize", 20, 30),
    )

    dataset = Dataset("NPP_season.csv", "RCP85", "RCP45", 0.2, ValidationSplit.QUANTILE_WITH_BINNING)
    emulator.fit(dataset.X_train, dataset.y_train, variable_names=dataset.X_variable_names, X_units=dataset.X_units,
                 y_units=dataset.y_units, validation_mask=dataset.validation_mask)
    return emulator.selected_validation_loss

if __name__ == '__main__':
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler())
    study.optimize(objective, n_trials=5)  # 50 évaluations au lieu de 10^5 en grid search
    print("Best hyperparameters:", study.best_params)
    print("Best accuracy:", study.best_value)