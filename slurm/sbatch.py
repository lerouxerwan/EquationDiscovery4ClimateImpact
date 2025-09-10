import os
import os.path as op
import socket
from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import Any

from utils.utils_bash_call import bash_call

LOCAL_COMPUTER = socket.gethostname() == 'IMT-MEE-20241210'
CURRENT_PATH = os.getcwd()


@dataclass
class Sbatch(ABC):
    """Call sbatch command for a specific python filepath that is called with some indices as arguments"""
    indices: list[int]

    def __post_init__(self):
        self.root = '/Odyssey/private/e23lerou/Documents/EquationDiscovery4ClimateImpact'

    @property
    @abstractmethod
    def filepath_from_root(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def sbatch_name(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def nb_cores(self) -> int:
        raise

    @property
    def setting_name(self) -> str:
        return f'{self.sbatch_name}_{'_'.join([str(i) for i in self.indices if i != 0])}'

    @property
    def python_exec(self):
        return f'python {self.root}/{self.filepath_from_root} {" ".join([str(i) for i in self.indices])}'

    @property
    def bash_filepath(self) -> str:
        short_setting_name = '_'.join([s[:3] for s in self.setting_name.split('_')])
        return op.join(self.root, 'slurm', 'bash_scripts', f'xp_{self.setting_name}', f'{short_setting_name}.sh')

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
        dirname = op.dirname(self.bash_filepath)
        if not LOCAL_COMPUTER:
            # Create folder if needed
            if not op.exists(dirname):
                os.makedirs(dirname, exist_ok=True)

        # Create empty bash file
        bash_call(f'touch {self.bash_filepath}', just_print=LOCAL_COMPUTER)

        if not LOCAL_COMPUTER:
            # Lines of the bash file
            lines = [
                '#!/bin/bash',
                f'export PATH="/Odyssey/private/e23lerou/shared_install/miniforge3/bin:$PATH"',
                'eval "$(mamba shell hook --shell bash)"',
                f'export PATH="/Odyssey/private/e23lerou/shared_install/julia-1.11.6/bin:$PATH"',
                f'export JULIA_DEPOT_PATH="/Odyssey/private/e23lerou/shared_install/.julia"',
                'export PYTHONPATH="${PYTHONPATH}:/Odyssey/private/e23lerou/Documents/EquationDiscovery4ClimateImpact"',
                f'mamba activate /Odyssey/private/e23lerou/shared_install/venv',
                self.python_exec,
                # f'rm {self.bash_filepath}'
            ]
            # Write lines inside the bash file
            with open(self.bash_filepath, 'w') as f:
                for line in lines:
                    f.write(f'{line}\n')

        # Make the bash file executable
        bash_call(f'chmod +x {self.bash_filepath}', just_print=LOCAL_COMPUTER)
        # Run bash file
        command = (f'sbatch '
                   f'-c {self.nb_cores} '
                   f'--time=2-00:00:00 '
                   f'--nodelist=sl-mee-br-111 '
                   f'-o {dirname}/%a.out '
                   f'{self.bash_filepath}')
        bash_call(command, just_print=LOCAL_COMPUTER)
