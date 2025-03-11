import numpy as np
from matplotlib import pyplot as plt

from tests.utils_tests_dataset import load_X_and_y_for_test


def apply_data_augmentation(X: np.ndarray, y: np.ndarray, data_augmentation_ratio: int, data_augmentation_sigma: float) -> tuple[np.ndarray, np.ndarray]:
    y_augmented = np.tile(y, data_augmentation_ratio)
    X_augmented = np.tile(X, (data_augmentation_ratio, 1))
    size = [X.shape[0] * (data_augmentation_ratio - 1), X.shape[1]]
    noise = np.concat([np.zeros(X.shape), np.random.normal(0.0, data_augmentation_sigma, size)])
    X_augmented += noise
    return X_augmented, y_augmented



def main_visualize_data_augmentation():
    ax = plt.gca()
    X, y = load_X_and_y_for_test()
    ax.plot(X[:, 0], label='original')
    for sigma in [1.0, 2.0, 4.0]:
        X_augmented, _ = apply_data_augmentation(X, y, 2, sigma)
        ax.plot(X_augmented[len(X):, 0], label=f'sigma={sigma}')
    ax.legend()
    plt.show()

if __name__ == '__main__':
    main_visualize_data_augmentation()