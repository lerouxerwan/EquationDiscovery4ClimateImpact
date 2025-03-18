import os
import os.path as op
from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import Union

from utils.utils_bash_call import bash_call
from utils.utils_path import CURRENT_PATH
from utils.utils_run import LOCAL_COMPUTER


@dataclass
class Sbatch(ABC):
    """Call sbatch command for a specific python filepath that is called with some indices as arguments"""
    indices: list[Union[int, float]]

    @property
    @abstractmethod
    def filepath(self) -> str:
        pass

    @property
    @abstractmethod
    def nb_cores(self) -> int:
        pass

    @property
    @abstractmethod
    def setting_name(self) -> str:
        pass

    @property
    def root(self) -> str:
        return "/homes/e23lerou/Documents/EquationDiscovery4ClimateImpact"

    @property
    def bash_filepath(self) -> str:
        job_name = '_'.join([s[:2] for s in self.setting_name.split('_')])
        return op.join(CURRENT_PATH, f'xp_{self.setting_name}', f'{job_name}.sh')


    @property
    def python_exec(self):
        return f'{self.root}/venv2/bin/python {self.root}/{self.filepath} {" ".join([str(i) for i in self.indices])}'

    @property
    def job_already_run(self) -> bool:
        """A job has already been run if a "out" file was generated and the "sh" file is not there.
         Indeed, if job terminated well, the bash file should have been deleted"""
        if op.exists(self.bash_filepath):
            files = os.listdir(op.dirname(self.bash_filepath))
            has_out_file = any([file.endswith('.out') for file in files])
            has_sh_file = any([file.endswith('.sh') for file in files])
            return has_out_file and (not has_sh_file)
        else:
            return False

    def run(self):
        if self.job_already_run:
            print(f'already done {self.setting_name}')
        else:
            print(f'sbatch {self.setting_name}')
            self._run()

    def _run(self):
        # Create folder if needed
        dirname = op.dirname(self.bash_filepath)
        if not op.exists(dirname):
            os.makedirs(dirname, exist_ok=True)
        # Check that this job has not yet been run

        # Create empty bash file
        bash_call(f'touch {self.bash_filepath}')
        # Lines of the bash file
        lines = [
            '#!/bin/bash',
            'source ~/.bashrc',
            f'cd {self.root}',
            self.python_exec,
            f'rm {self.bash_filepath}'
        ]
        # Write lines inside the bash file
        with open(self.bash_filepath, 'w') as f:
            for line in lines:
                f.write(f'{line}\n')
        # Run bash file
        command = (f'sbatch '
                   f'-p Odyssey '
                   f'-c {self.nb_cores} '
                   f'--time=2-00:00:00 '
                   f'-o {dirname}/%a.out'
                   f' {self.bash_filepath}')
        print(command)
        if not LOCAL_COMPUTER:
            bash_call(f"chmod +x {self.bash_filepath}")
            bash_call(command)
