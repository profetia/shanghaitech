from typing import Dict, List, Tuple
from fastapi import APIRouter
import horras_backend.model.geometry.queries as queries
from horras_backend.model.heatmap.ranges import SampleGroup

router = APIRouter()

def convert_to_dict(data: List[Dict], field: str, suffix: str) -> Dict[Tuple[int, int], Dict[str, int]]:
    return {(item[f"{field}_lat{suffix}"], item[f"{field}_lng{suffix}"]): {
        "id": item[f"{field}_district"], 
        "count": item["_count"]["_all"]
    } for item in data}

@router.post("/")
async def geometry(query: SampleGroup) -> List[Dict[str, int]]:
    date_lower, date_upper = query.date_range
    clock_lower, clock_upper = query.clock_range
    sample = query.sample

    if not sample:
        suffix = ""
    else:
        suffix = "_group"
    
    departure_records: List[Dict] = await queries.find_records_group(
        date_lower, date_upper, clock_lower, clock_upper, "departure", suffix
    )

    departure_dict: Dict[Tuple[int, int], Dict[str, int]] = convert_to_dict(departure_records, 'starting', suffix)
    
    arrive_records: List[Dict] = await queries.find_records_group(
        date_lower, date_upper, clock_lower, clock_upper, "arrive", suffix
    )
    arrive_dict: Dict[Tuple[int, int], Dict[str, int]] = convert_to_dict(arrive_records, 'dest', suffix)

    final_dict: Dict[Tuple[int, int], Dict[str, int]] = {}
    for key in set(departure_dict.keys()).union(set(arrive_dict.keys())):
        final_dict[key] = {
            'id': arrive_dict[key]['id'] if key in arrive_dict else departure_dict[key]['id'],
            'start_num': departure_dict[key]['count'] if key in departure_dict else 0,
            'dest_num': arrive_dict[key]['count'] if key in arrive_dict else 0,
        }

    records: List[Dict[str, int]] = [
        {
            'lat': key[0], 
            'lng': key[1], 
            **value, 
        } for key, value in final_dict.items()
    ]

    return records