import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from rqfactor import *
from rqfactor.extension import *
from rqdatac import * 
from mina_factor.mina_scope import *

rqdatac.init(uri='tcp://rice:rice@192.168.10.40:16019')

class StockDataset(Dataset):
    def __init__(self, df, window=30, step=1):
        self.df = df
        self.window = window
        self.step = step
        self.trading_days = sorted(df.index.get_level_values('date').unique())
        self.num_samples = (len(self.trading_days) - window) // step + 1

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start_idx = idx * self.step
        end_idx = start_idx + self.window
        date_window = self.trading_days[start_idx:end_idx]
        
        window_data = self.df.loc[date_window]
        data_images = []
        labels = []
        removed_stocks = []

        grouped = window_data.groupby(level='order_book_id')
        for stock_id, stock_data in grouped:
            stock_data = stock_data.values
            if stock_data.shape[0] >= self.window:
                data_images.append(stock_data[:, :-1])
                labels.append(stock_data[-1, -1])
            else:
                removed_stocks.append(stock_id)
        
        if data_images:
            data_images = np.array(data_images).transpose(0, 2, 1)
            labels = np.array(labels)
            return torch.tensor(data_images, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32), removed_stocks
        else:
            return None

def dynamic_collate_fn(batch):
    images, labels, nonrealize = batch[0]  # batch_size=1, 所以直接取第一个元素
    return images, labels, nonrealize

def create_dynamic_dataloader(df, window=30, step=1, shuffle=True):
    dataset = StockDataset(df, window, step)
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
    stop = '20240801'
    rollback = get_previous_trading_date(stop, 504 + 63*8 + 5*2 + 30 -2)
    universe = get_universe('20140101', stop)
    future = get_future(universe=universe, period=5, start=rollback, stop=stop)
    future = future.stack().to_frame('future').sort_index()
    data = get_price(order_book_ids=universe, fields=['open', 'high', 'low', 'close', 'volume', 'num_trades'],
                     frequency='1d', adjust_type='post_volume', start_date=rollback, end_date=stop).swaplevel('date', 'order_book_id')
    df = pd.merge(data.reset_index(), future.reset_index(), on=['order_book_id', 'date'], how='inner')
    df = df.set_index(['date', 'order_book_id']).sort_index()
    df.to_parquet('dataset.parquet')

