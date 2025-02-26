from slurm.sbatch import Sbatch


class SBatchExample(Sbatch):

    @property
    def filepath(self) -> str:
        return 'slurm/example/main_example.py'

    @property
    def nb_cores(self) -> int:
        return 1

    @property
    def setting_name(self) -> str:
        return f'{self.indices[0]}Index'

