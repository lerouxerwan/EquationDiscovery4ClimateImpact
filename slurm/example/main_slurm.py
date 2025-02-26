from slurm.example.sbatch_example import SBatchExample


def main_run_multiple_sbatch():
    for nb_features in  [0, 1]:
        indices = [nb_features]
        sbatch = SBatchExample(indices)
        sbatch.run()


if __name__ == '__main__':
    main_run_multiple_sbatch()
