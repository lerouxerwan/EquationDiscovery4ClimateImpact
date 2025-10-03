import os
import os.path as op

#  Path to root folder
ROOT = op.dirname(op.dirname(os.path.abspath(__file__)))

# Path to the current directory
CURRENT_PATH = os.getcwd()

#  Data parameters
DATA_PATH = op.join(ROOT, 'data')
DATASET_CSV_PATH = op.join(DATA_PATH, 'dataset')
RUN_PATH = op.join(DATA_PATH, 'run')
OPT_PATH = op.join(DATA_PATH, 'opt')
NESTED_CV_PATH = op.join(DATA_PATH, 'nested_cv')

#  Result parameters
RESULT_PATH = op.join(ROOT, 'results')


