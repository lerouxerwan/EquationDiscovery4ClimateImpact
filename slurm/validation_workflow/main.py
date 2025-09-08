from slurm.sbatch import Sbatch


class SbatchValidationWorkflow(Sbatch):

    @property
    def sbatch_name(self) -> str:
        return 'val_workflow'

    @property
    def filepath_from_root(self):
        return 'slurm/validation_workflow/main_for_bash_script.py'

    @property
    def nb_cores(self) -> int:
        return 8


def main():
    for n_iter in [10]:
        # for nb_top_hyperparameters in range(1, 11):
        for nb_top_hyperparameters in [3]:
            indices = [0, 0, n_iter, nb_top_hyperparameters, 0]
            sbatch = SbatchValidationWorkflow(indices)
            sbatch.run()


if __name__ == '__main__':
    main()