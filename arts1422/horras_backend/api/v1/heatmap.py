from typing import List, Dict
from fastapi import APIRouter
import horras_backend.model.heatmap.ranges as ranges
import horras_backend.model.heatmap.queries as queries
import horras_backend.model.heatmap.cheat as cheat

router = APIRouter()

@router.get("/")
async def headmap() -> Dict[str, List[List[int]]]:
    # clock_range = ranges.CLOCK_UPPER_BOUND - ranges.CLOCK_LOWER_BOUND
    # date_range = ranges.DATE_UPPER_BOUND - ranges.DATE_LOWER_BOUND
    # heatmap_values = [[0 for i in range(clock_range // 60)] for j in range(date_range)]
    # for date in range(date_range):
    #     for clock in range(0, clock_range, 60):
    #         print(f"{ranges.DATE_LOWER_BOUND + date}-{ranges.DATE_LOWER_BOUND + date+1}:{clock}-{clock + 60}")
    #         heatmap_values[date][clock // 60] = await queries.find_records_count(ranges.DATE_LOWER_BOUND + date, ranges.DATE_LOWER_BOUND + date + 1, clock, clock + 60)
    
    # target = [300, 301, 302, 303, 304, 305]
    # heatmap_values = [[0 for i in range(1440 // 60)] for j in range(len(target))]
    # for i, date in enumerate(target):
    #     for clock in range(0, 1440, 60):
    #             print(f"{date}-{date+1}:{clock}-{clock + 60}")
    #             heatmap_values[i][clock // 60] = await queries.find_records_count(date, date + 1, clock, clock + 60)
                
    heatmap_values = cheat.get_headmap()
    init_geometry = cheat.get_init_geometry()
    return {
        "heatmap": heatmap_values,
        "init_geometry": init_geometry
    }
