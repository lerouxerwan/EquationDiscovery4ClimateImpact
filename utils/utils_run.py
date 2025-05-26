import socket
from multiprocessing import cpu_count

random_seed = 42

MAX_NB_JOBS = cpu_count() - 1 # We keep one job for other tasks

LOCAL_COMPUTER = socket.gethostname() == "IMT-MEE-20241210"



