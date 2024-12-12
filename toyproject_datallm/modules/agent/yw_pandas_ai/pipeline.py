import logging
import time
from typing import Any, List, Optional
from base_logic_unit import BaseLogicUnit

class Pipeline:
    def __init__(self, steps: Optional[List[BaseLogicUnit]] = None):
        self.steps = steps or []
        self.logger = logging.getLogger("Pipeline")

    def add_step(self, logic: BaseLogicUnit):
        if not isinstance(logic, BaseLogicUnit):
            raise TypeError("Logic unit must be inherited from BaseLogicUnit.")
        self.steps.append(logic)

    def run(self, data: Any = None) -> Any:
        try:
            for index, logic in enumerate(self.steps):
                if logic.before_execution is not None:
                    logic.before_execution(data)

                self.logger.info(f"Executing Step {index}: {logic.__class__.__name__}")

                if logic.skip_if is not None and logic.skip_if():
                    self.logger.info(f"Skipping Step {index}")
                    continue

                start_time = time.time()
                data = logic.execute(data)
                execution_time = time.time() - start_time

                self.logger.info(f"Step {index} executed in {execution_time:.4f} seconds")

                if logic.on_execution is not None:
                    logic.on_execution(data)

        except Exception as e:
            self.logger.error(f"Pipeline failed on step {index}: {e}")
            raise e

        return data