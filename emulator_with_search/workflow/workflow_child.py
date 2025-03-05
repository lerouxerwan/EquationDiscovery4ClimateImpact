from dataclasses import dataclass
from typing import Any

from emulator_with_search.search_dir.search_dir import SearchDir
from emulator_with_search.workflow.workflow import Workflow


@dataclass
class WorkflowChild(Workflow):
    params_child_search: dict[str, Any]
    search_dir_parent: str = None

    def __post_init__(self):
        # If search_dir parent is not specified, we set it as equal to the search_dir that minimizes the loss
        if self.search_dir_parent is None:
            raise NotImplementedError

    @property
    def params_emulator(self) -> dict[str, Any]:
        params_emulator = SearchDir(self.search_dir_parent).best_params
        params_emulator.update(self.params_child_search)
        return params_emulator