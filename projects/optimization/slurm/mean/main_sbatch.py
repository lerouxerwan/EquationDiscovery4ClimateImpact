from projects.optimization.slurm.sbatch import Sbatch


class SBatchWorkflowMean(Sbatch):

    @property
    def filepath(self) -> str:
        return 'projects/optimization/slurm/mean/main_workflow.py'

    @property
    def nb_cores(self) -> int:
        return 32

    @property
    def setting_name(self) -> str:
        return 'workflow_mean_' + str(self.indices[0])

def main_run_multiple_sbatch():
    for index in range(2):
        SBatchWorkflowMean(indices=[index]).run()


if __name__ == '__main__':
    main_run_multiple_sbatch()