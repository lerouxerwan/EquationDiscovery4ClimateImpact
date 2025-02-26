from slurm.sbatch import Sbatch


class SBatchWorkflowV1(Sbatch):

    @property
    def filepath(self) -> str:
        return 'slurm/workflow_v1/main_workflow_v1.py'

    @property
    def nb_cores(self) -> int:
        return 1

    @property
    def setting_name(self) -> str:
        return f'{self.indices[0]}Features'

