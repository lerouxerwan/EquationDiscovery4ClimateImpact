from slurm.sbatch import Sbatch


class SbatchGaussian(Sbatch):

    @property
    def sbatch_name(self) -> str:
        return 'g'

    @property
    def filepath_from_root(self):
        return 'slurm/nested_cv/main_for_bash_script.py'

    @property
    def nb_cores(self) -> int:
        return 12

    @property
    def server_name(self) -> str:
        return '--nodelist=sl-mee-br-112'


def main_g():
    for index in [0, 1, 2][:]:
        indices = [index]
        sbatch = SbatchGaussian(indices)
        sbatch.run()


if __name__ == '__main__':
    main_g()