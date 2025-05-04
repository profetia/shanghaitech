import pandas as pd
import datetime

def get_date_and_time(df, name):
    full_time = df[name].to_pydatetime()
    date = (full_time - datetime.datetime(2017, 1, 1)).days
    time = full_time.hour * 60 + full_time.minute
    return date, time

def parse_time(df):
    df['departure_date'], df['departure_time'] = get_date_and_time(df, 'departure_time')
    df['arrive_date'], df['arrive_time'] = get_date_and_time(df, 'arrive_time')
    return df

def refine():
    data: pd.DataFrame = pd.read_pickle('./data.pkl')
    data.dropna(inplace=True)
    data.reset_index(inplace=True)
    data.index.name = 'id'

    print('parse time')
    data = data.apply(lambda x: parse_time(x), axis=1)

    print('save')
    data.to_pickle('./data_refined.pkl')
    data.to_csv('./data_refined.csv')


if __name__ == '__main__':
    refine()