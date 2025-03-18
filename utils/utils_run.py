import socket
from multiprocessing import cpu_count

random_seed = 42

NB_CORES = cpu_count() - 1

LOCAL_COMPUTER = socket.gethostname().startswith('IMT-MEE')

