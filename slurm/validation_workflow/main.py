from slurm.sbatch import Sbatch


class SbatchValidationWorkflow(Sbatch):

    @property
    def filepath_from_root(self):
        return 'slurm/validation_workflow/main_for_bash_script.py'

    @property
    def setting_name(self) -> str:
        return f'val_workflow_{'_'.join([str(i) for i in self.indices])}'


def main():
    for n_iter in [10]:
        # for nb_top_hyperparameters in range(1, 11):
        for nb_top_hyperparameters in [3]:
            indices = [0, 0, n_iter, nb_top_hyperparameters, 0]
            sbatch = SbatchValidationWorkflow(indices)
            sbatch.run()


if __name__ == '__main__':
    main()