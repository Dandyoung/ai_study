import pandas as pd
from typing import Any
from base_logic_unit import BaseLogicUnit

class SplitDataFrameByMinutes(BaseLogicUnit):
    def __init__(self, minutes: int, **kwargs):
        super().__init__(**kwargs)
        self.minutes = minutes

    def execute(self, input: pd.DataFrame, **kwargs) -> Any:
        if 'timestamp' not in input.columns:
            raise ValueError("DataFrame must have a 'timestamp' column.")
        
        input['timestamp'] = pd.to_datetime(input['timestamp'])
        result = []
        grouped = input.set_index('timestamp').resample(f'{self.minutes}T')
        for _, group in grouped:
            result.append(group.reset_index())

        return result