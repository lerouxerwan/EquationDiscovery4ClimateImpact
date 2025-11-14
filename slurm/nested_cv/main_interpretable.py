from dataclasses import dataclass

from slurm.sbatch import Sbatch


@dataclass
class SbatchInterpretable(Sbatch):
    server_number: int = 112

    @property
    def sbatch_name(self) -> str:
        return 'int'

    @property
    def filepath_from_root(self):
        return 'slurm/nested_cv/main_for_bash_script.py'

    @property
    def server_name(self) -> str:
        return f'--nodelist=sl-mee-br-{self.server_number}'


def main_marginal():
    index1 = 0
    for index2 in [0, 1, 2][:]:
        for index3 in [0, 1, 2][:]:
            indices = [index1, index2, index3]
            sbatch = SbatchInterpretable(indices)
            sbatch.run()


def main_marginal_and_random():
    index1 = 2
    for index2 in [0, 1, 2][:]:
        for index3 in [0, 1, 2][:]:
            indices = [index1, index2, index3]
            server_number = 113
            if index2 == 0:
                if index3 == 0:
                    server_number = 111
                if index3 == 1:
                    server_number = 112
            sbatch = SbatchInterpretable(indices, server_number=server_number)
            sbatch.run()


def main_random():
    index1 = 1
    for index2 in [0, 1, 2][:]:
        for index3 in [0, 1, 2][:]:
            SbatchInterpretable([index1, index2, index3]).run()

if __name__ == '__main__':
    # main_marginal()
    # main_random()
    main_marginal_and_random()