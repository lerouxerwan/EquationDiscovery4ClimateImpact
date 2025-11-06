
from slurm.sbatch import Sbatch


class SbatchNestedCV(Sbatch):

    @property
    def sbatch_name(self) -> str:
        return 'bay'

    @property
    def filepath_from_root(self):
        return 'projects/gaussian_True/main_opt_bayesian.py'

    @property
    def nb_cores(self) -> int:
        return 4

    @property
    def server_name(self) -> str:
        return '--nodelist=sl-mee-br-113'


def main_validation_size():
    index1 = 1
    for index2 in [0, 1, 2][:]:
        for index3 in [0, 1, 2][:]:
            indices = [index1, index2, index3]
            sbatch = SbatchNestedCV(indices)
            sbatch.run()


if __name__ == '__main__':
    main_validation_size()