from slurm.workflow_v1.sbatch_workflow_v1 import SBatchWorkflowV1


def main_run_multiple_sbatch():
    for nb_features in  [3, 4, 5]:
        indices = [nb_features]
        sbatch = SBatchWorkflowV1(indices)
        sbatch.run()


if __name__ == '__main__':
    main_run_multiple_sbatch()
