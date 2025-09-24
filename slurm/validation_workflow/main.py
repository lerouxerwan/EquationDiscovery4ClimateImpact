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
        return 20


def main_validation_size():
    for n_iter in [100]:
        for nb_top_hyperparameters in [5]:
            for index in [1, 2]:
                indices = [0, 0, n_iter, nb_top_hyperparameters, index]
                sbatch = SbatchValidationWorkflow(indices)
                sbatch.run()


def main_max_depth():
    for n_iter in [100]:
        for nb_top_hyperparameters in [5]:
            # for index in range(2, 10):
            for max_depth in [5, 6][:]:
                indices = [0, 0, n_iter, nb_top_hyperparameters, max_depth]
                sbatch = SbatchValidationWorkflow(indices)
                sbatch.run()


if __name__ == '__main__':
    # main_validation_size()
    main_max_depth()