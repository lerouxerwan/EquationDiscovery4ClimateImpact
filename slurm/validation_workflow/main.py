from slurm.sbatch import Sbatch


class SbatchValidationWorkflow(Sbatch):

    @property
    def sbatch_name(self) -> str:
        return 'val'

    @property
    def filepath_from_root(self):
        return 'slurm/validation_workflow/main_for_bash_script.py'

    @property
    def nb_cores(self) -> int:
        return 40


def main_validation_size():
    for n_iter in [1000]:
        for nb_top_hyperparameters in [5]:
            for index in [2, 3, 4, 5, 6, 7, 8][:1]:
                indices = [0, 0, n_iter, nb_top_hyperparameters, index]
                sbatch = SbatchValidationWorkflow(indices)
                sbatch.run()


if __name__ == '__main__':
    main_validation_size()