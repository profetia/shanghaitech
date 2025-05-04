from typing import Dict, List
from horras_backend.database import prisma

async def find_records_group(date_lower: int, 
                            date_upper: int, 
                            clock_lower: int, 
                            clock_upper: int, 
                            kind: str, suffix: str) -> List[Dict]:
    if kind == "departure":
        date: str = 'departure_date'
        time: str = 'departure_time'
        group: str = ['starting_district', f'starting_lng{suffix}', f'starting_lat{suffix}']
    else:
        date: str = 'arrive_date '
        time: str = 'arrive_time'
        group: str = ['dest_district', f'dest_lng{suffix}', f'dest_lat{suffix}']

    return await prisma.horras.group_by(by=group, where={
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
            }
        ]
    }, count=True)

