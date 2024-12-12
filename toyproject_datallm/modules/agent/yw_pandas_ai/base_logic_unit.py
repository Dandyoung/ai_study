from abc import ABC, abstractmethod
from typing import Any

class BaseLogicUnit(ABC):
    def __init__(self, skip_if=None, on_execution=None, before_execution=None):
        self.skip_if = skip_if
        self.on_execution = on_execution
        self.before_execution = before_execution

    @abstractmethod
    def execute(self, input: Any, **kwargs) -> Any:
        raise NotImplementedError("execute method is not implemented.")