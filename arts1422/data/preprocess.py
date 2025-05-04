import pandas as pd
import numpy as np
import numba as nb
import json
import time
import datetime
from multiprocessing import Pool
import os

np.seterr(divide='ignore', invalid='ignore')

@nb.njit(fastmath=True)
def in_polygon_nb(points:np.ndarray,poly:np.ndarray)->np.ndarray:
    n = len(poly)
    res = np.zeros(len(points), dtype=np.bool_)
    for i in range(len(points)):
        p = points[i]
        j = n - 1
        for k in range(n):
            if (poly[k,1] - p[1]) * (poly[j,1] - p[1]) < 0 and p[0] < (poly[j,0] - poly[k,0]) * (p[1] - poly[k,1]) / (poly[j,1] - poly[k,1]) + poly[k,0]:
                res[i] = not res[i]
            j = k
    return res

def in_polygon_np(points:np.ndarray,poly:np.ndarray)->np.ndarray:
    p = np.expand_dims(poly, axis=0) - np.expand_dims(points, axis=1)
    d = p[:,1:,:] - p[:,:-1,:]
    return np.sum((p[:,:-1,1] * p[:,1:,1]<=0) & (p[:,:-1,0] - p[:,:-1,1] * d[:,:,0] / d[:,:,1] >0), axis=1) & 1 == 1

in_polygon = in_polygon_nb


def get_date_and_time(df, name):
    full_time = df[name].to_pydatetime()
    date = (full_time - datetime.datetime(2017, 1, 1)).days
    time = full_time.hour * 60 + full_time.minute
    return date, time

def parse_time(df):
    df['departure_date'], df['departure_time'] = df['departure_time'].day_of_year, df['departure_time'].hour * 60 + df['departure_time'].minute
    df['arrive_date'], df['arrive_time'] = df['arrive_time'].day_of_year, df['arrive_time'].hour * 60 + df['arrive_time'].minute
    return df

def run(i:int):

    data = pd.read_csv(f'./taxi/dwv_order_make_haikou_{i}.csv', sep='\t')
    data.rename(columns={c:c[24:] for c in data.columns}, inplace=True)
    data['arrive_time'].replace('0000-00-00 00:00:00',None, inplace=True)
    data['departure_time'] = pd.to_datetime(data['arrive_time'])
    data['arrive_time'] = data['departure_time'] + pd.to_timedelta(data['normal_time'], 'm')
    data = data[['county', 'traffic_type', 'start_dest_distance', 'departure_time', 'arrive_time','normal_time', 'product_1level', 'dest_lng', 'dest_lat', 'starting_lng', 'starting_lat']]

    for county_code in range(460105,460109):
        geo_json = json.load(open(f'./geojson/{county_code}.json',encoding='utf-8'))
        for j, feature in enumerate(geo_json['features']):
            start_time = time.time()
            polygons = np.array(feature['geometry']['coordinates'][0])
            in_poly = in_polygon(data[['starting_lng','starting_lat']].to_numpy(),polygons)
            data.loc[in_poly,'starting_district'] = f'{county_code}{j:02d}'
            print(f'dwv_order_make_haikou_{i} {county_code}{j:02d} {feature["properties"]["name"]:<5} {time.time() - start_time:.2f}s, {np.sum(in_poly)} records')
            start_time = time.time()
            in_poly = in_polygon(data[['dest_lng','dest_lat']].to_numpy(),polygons)
            data.loc[in_poly,'dest_district'] = f'{county_code}{j:02d}'
            print(f'dwv_order_make_haikou_{i} {county_code}{j:02d} {feature["properties"]["name"]:<5} {time.time() - start_time:.2f}s, {np.sum(in_poly)} records')
    
    data = data.dropna()
    data = data.apply(parse_time, axis=1)
    data.to_pickle(f'./data_{i}.pkl')

if __name__ == '__main__':

    # target = [8]
    
    # pool = Pool(len(target))
    # pool.map(run, target)
    # pool.close()
    # pool.join()

    print('merge')

    pd.concat([pd.read_pickle(f'./data_{i}.pkl') for i in range(1,9)]).to_csv('./data.csv', index=False)

    print('done')

    # for i in range(1,9):
    #    os.remove(f'./data_{i}.pkl')
