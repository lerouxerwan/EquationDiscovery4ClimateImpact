from projects.optimization.slurm.sbatch import Sbatch


class SBatchWorkflowV1(Sbatch):

    @property
    def filepath(self) -> str:
        return 'projects/optimization/slurm/v1/main_workflow.py'

    @property
    def nb_cores(self) -> int:
        return 16

    @property
    def setting_name(self) -> str:
        return f'workflowV1_scaling={self.indices[0]}'

def main_run_multiple_sbatch():
    for scaling in  [1.5, 2., 4.]:
        SBatchWorkflowV1([scaling]).run()


if __name__ == '__main__':
    main_run_multiple_sbatch()