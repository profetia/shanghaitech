from typing import Dict, List
from horras_backend.database import prisma

async def find_records_topo(date_lower: int, 
                            date_upper: int, 
                            clock_lower: int, 
                            clock_upper: int, 
                            target_nodes: List[int],
                            kind: str) -> List[Dict]:
    if kind == "departure":
        date: str = 'departure_date'
        time: str = 'departure_time'
        group: str = 'starting_district'
    else:
        date: str = 'arrive_date '
        time: str = 'arrive_time'
        group: str = 'dest_district'

    return await prisma.horras.group_by(by=['starting_district', 'dest_district'], where={
        'AND': [
            {
                date: {
                    'gte': date_lower,
                    'lt': date_upper
                }
            },
            {
                time: {
                    'gte': clock_lower,
                    'lt': clock_upper
                }
            },
            {
                group: {
                    'in': target_nodes
                }
            }
        ]
    }, count=True)