from typing import List
from horras_backend.model.heatmap.ranges import TimeRange

class Selection(TimeRange):
    nodes: List[int]