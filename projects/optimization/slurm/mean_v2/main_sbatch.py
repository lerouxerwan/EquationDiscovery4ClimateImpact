from projects.optimization.slurm.sbatch import Sbatch


class SBatchWorkflowMeanV2(Sbatch):

    @property
    def filepath(self) -> str:
        return 'projects/optimization/slurm/mean_v2/main_workflow.py'

    @property
    def nb_cores(self) -> int:
        return 16

    @property
    def setting_name(self) -> str:
        return 'workflow_mean_v2_' + str(self.indices[0])

def main_run_multiple_sbatch():
    for index in range(4):
        SBatchWorkflowMeanV2(indices=[index]).run()


if __name__ == '__main__':
    main_run_multiple_sbatch()