from typing import Any
import pandas as pd
from base_logic_unit import BaseLogicUnit

class AggregateStatistics(BaseLogicUnit):
    def execute(self, input: Any, **kwargs) -> Any:
        if not isinstance(input, list):
            raise ValueError("Input should be a list of DataFrames.")
        
        aggregated_results = []
        for df in input:
            stats = {
                "min": df["value"].min(),
                "max": df["value"].max(),
                "mean": df["value"].mean()
            }
            aggregated_results.append(stats)
        
        return aggregated_results