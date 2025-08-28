import numpy as np

from emulator.emulator import Emulator
from tests.data.utils_tests_dataset import load_X_and_y_for_test
from utils.utils_log import log_info


def main():
    emulator = Emulator(niterations=1)
    X, y = load_X_and_y_for_test()
    log_info(f'{X.sum()} {y.sum()}')
    emulator.fit(X, y)
    #  Assert that the fit of the emulator is deterministic
    loss_list = emulator.loss_list
    sum_loss_list = float(sum(loss_list))
    print_line = f'{sum_loss_list} {loss_list}'
    log_info(print_line)
    emulator.remove_folder()
    np.testing.assert_almost_equal(sum_loss_list, 35698078.93297232)

if __name__ == '__main__':
    main()