from typing import Tuple
from pydantic import BaseModel

DATE_LOWER_BOUND = 119
DATE_UPPER_BOUND = 305
CLOCK_LOWER_BOUND = 0
CLOCK_UPPER_BOUND = 1440

class TimeRange(BaseModel):
    date_range: Tuple[int, int]
    clock_range: Tuple[int, int]

class SampleGroup(TimeRange):
    sample: bool # false = ungrouped, true = grouped