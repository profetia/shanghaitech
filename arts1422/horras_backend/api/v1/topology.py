from typing import Dict, List, Tuple
from fastapi import APIRouter
from horras_backend.model.topology.selection import Selection
from horras_backend.model.topology.queries import find_records_topo

router = APIRouter()

def convert_to_dict(records: List[Dict], kind: str) -> Dict[Tuple[int, int], int]:
    if kind == "departure":
        return {
            (record["starting_district"], record["dest_district"]): record["_count"]["_all"] \
                for record in records
        }
    else:
        return {
            (record["dest_district"], record["starting_district"]): record["_count"]["_all"] \
                for record in records
        }


@router.post("/")
async def topology(query: Selection) -> Dict[str, List[Dict]]:
    date_lower, date_upper = query.date_range
    clock_lower, clock_upper = query.clock_range
    target_nodes = query.nodes

    departure_records: List[Dict] = await find_records_topo(date_lower, date_upper, 
        clock_lower, clock_upper, target_nodes, "departure")
    departure_dict: Dict[Tuple[int, int], int] = convert_to_dict(departure_records, "departure")

    arrive_records: List[Dict] = await find_records_topo(date_lower, date_upper,
        clock_lower, clock_upper, target_nodes, "arrive")
    arrive_dict: Dict[Tuple[int, int], int] = convert_to_dict(arrive_records, "arrive")

    final_dict: Dict[Tuple[int, int], int] = {}
    for key in set(departure_dict.keys()).union(set(arrive_dict.keys())): 
        x, y = key
        final_dict[(x, y)] = {
            "xy_num": departure_dict.get((x, y), 0),
            "yx_num": arrive_dict.get((x, y), 0)
        }

    records: Dict[str, List[Dict]] = {
        "nodes": list(set([x for x, y in final_dict.keys()] + [y for x, y in final_dict.keys()])),
        "edges": [{"x": x, "y": y, **final_dict[(x, y)]} for x, y in final_dict.keys()]
    }

    return records