from dataclasses import dataclass
from typing import Optional

from slurm.sbatch import Sbatch


@dataclass
class SbatchInterpretable(Sbatch):
    server_numbers: Optional[list[int]] = None

    def __post_init__(self):
        super().__post_init__()
        if self.server_numbers is None:
            self.server_numbers = [112]

    @property
    def sbatch_name(self) -> str:
        return 'int'

    @property
    def filepath_from_root(self):
        return 'slurm/nested_cv/main_for_bash_script.py'

    @property
    def server_name(self) -> str:
        server_numbers_as_str = ','.join([f'sl-mee-br-{server_number}' for server_number in self.server_numbers])
        return f'--nodelist={server_numbers_as_str}'


def main_random():
    for index2, server_number in zip([0, 1, 2], [111, 112, 113]):
        for index3 in [0, 1][:]:
            SbatchInterpretable([0, index2, index3], server_numbers=[server_number], nb_cores=16).run()


if __name__ == '__main__':
    main_random()