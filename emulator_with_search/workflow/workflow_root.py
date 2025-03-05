from dataclasses import dataclass
from typing import Any

from emulator_with_search.workflow.workflow import Workflow


@dataclass
class WorkflowRoot(Workflow):
    params_root_search: dict[str, Any]

    @property
    def params_emulator(self) -> dict[str, Any]:
        return self.params_root_search

