from slurm.sbatch import Sbatch


class SbatchNestedCV(Sbatch):

    @property
    def sbatch_name(self) -> str:
        return 'b'

    @property
    def filepath_from_root(self):
        return 'slurm/nested_cv/main_for_bash_script.py'

    @property
    def nb_cores(self) -> int:
        return 1

    @property
    def server_name(self) -> str:
        return '--nodelist=sl-mee-br-112'


def main_validation_size():
    for index1 in [0, 1, 2][:]:
        for index2 in [0, 1, 2][:]:
            for index3 in [0, 1, 2][:]:
                indices = [index1, index2, index3]
                sbatch = SbatchNestedCV(indices)
                sbatch.run()


if __name__ == '__main__':
    main_validation_size()