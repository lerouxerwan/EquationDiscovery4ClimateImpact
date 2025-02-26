import sys
import time

from projects.paper.utils_paper import filename_dataset_paper
from projects.utils_workflow import workflow
from utils.utils_log import log_info


def main_example():
    indices = [int(s) for s in sys.argv[1:]]
    log_info(f'indices={indices}')


if __name__ == '__main__':
    main_example()


