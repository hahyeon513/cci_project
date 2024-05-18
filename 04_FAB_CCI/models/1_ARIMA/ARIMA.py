#%%
import pandas as pd
import numpy as np

class ARIMA:
    def __init__(self, target_data_path, factor_data_path):
        self.target_df = pd.read_csv(str(target_data_path), index_col='date', parse_dates=True)
        self.factor_df = pd.read_csv(str(factor_data_path), index_col='date', parse_dates=True)
        self.df = self.target_df.join(self.factor_df, how='inner')