import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from rqfactor import *
from rqfactor.extension import *
from rqdatac import * 
from mina_factor.mina_scope import *
import quool

rqdatac.init(uri='tcp://rice:rice@192.168.10.40:16019')
quotes_10min = quool.Factor("./data/quotes_10min", code_level="order_book_id", date_level="date")

class StockDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.unique_dates = sorted(df.index.get_level_values('date').strftime('%Y-%m-%d').unique())
        
    def __len__(self):
        return len(self.unique_dates)

    def __getitem__(self, idx):
        date_str = self.unique_dates[idx]
        date_data = self.df.loc[self.df.index.get_level_values('date').strftime('%Y-%m-%d') == date_str]

        data_images = []
        labels = []

        grouped = date_data.groupby(level='order_book_id')
        for stock_id, stock_data in grouped:
            stock_data = stock_data.values
            data_images.append(stock_data[:, :-1]) 
            labels.append(stock_data[-1, -1]) 

        if data_images:
            data_images = np.array(data_images).transpose(0, 2, 1)
            labels = np.array(labels)
            return torch.tensor(data_images, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)
        else:
            return None

def dynamic_collate_fn(batch):
    images, labels = batch[0]  # batch_size=1, 所以直接取第一个元素
    return images, labels

def create_dynamic_dataloader(df, shuffle=True):
    dataset = StockDataset(df)
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=dynamic_collate_fn, shuffle=shuffle)
    return dataloader

def get_future(
    period: int = 1, 
    start: str | pd.Timestamp = None,
    stop: str | pd.Timestamp = None,
    universe: list = None
):
    if stop is not None:
        stop = get_next_trading_date(stop, period + 1)
    price = rqdatac.get_vwap(universe, start_date=start, end_date=stop, frequency='1d').unstack('order_book_id')
    future = price.shift(-1 - period) / price.shift(-1) - 1
    future = future.dropna(axis=0, how='all')
    future = future.replace([np.inf, -np.inf], np.nan)
    return future.squeeze()

def get_universe(start, stop):
    start = start
    stop =  stop
    insts = all_instruments('CS')
    universe = sorted(insts[((insts['de_listed_date'] == '0000-00-00') | (insts['de_listed_date'] > start))&(insts['listed_date'] <= stop)].order_book_id.tolist())

    for i in get_share_transformation().itertuples():
        if i.event == "code_change":
            if i.predecessor in universe:
                universe.remove(i.predecessor)
            else:
                print(f"Warning: Code {i.predecessor} not found in universe and cannot be removed.")
    return universe

if __name__ == '__main__':
    stop = '20240901'
    rollback = get_previous_trading_date(stop, 504 + 63*7 + 1*2 + 20 -1)
    universe = get_universe('20140101', stop)

    future = get_future(
        universe=universe, 
        period=1, 
        start=rollback, 
        stop=stop
        ).stack()
    future.name = 'future_1d'

    vwap = rqdatac.get_vwap(
        universe, 
        start_date=rollback, 
        end_date=stop, 
        frequency='10m'
        )
    vwap.name = 'vwap'

    data = get_price(
        order_book_ids=universe, 
        fields=['open', 'high', 'low', 'close', 'volume', 'num_trades'],
        frequency='10m', 
        adjust_type='post_volume', 
        start_date=rollback, 
        end_date=stop, 
        expect_df=True
        )
    df = pd.merge(data, vwap, on=['datetime','order_book_id'], how='inner')

    df['date'] = df.index.get_level_values('datetime').normalize()
    df = df.reset_index().sort_values(['date', 'order_book_id'])
    future = future.reset_index().sort_values(['date', 'order_book_id'])
    df = pd.merge(df, future, on=['date', 'order_book_id'], how='inner')
    df = df.drop(columns=['date']).rename(columns={'datetime': 'date'}).set_index(['date', 'order_book_id']).sort_index()
    quotes_10min.update(df)

