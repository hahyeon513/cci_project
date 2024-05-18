#%%
import pandas as pd
import numpy as np

# 데이터 로드
data = pd.read_csv('C:/Users/Yoon Ha Hyeon/Desktop/CCI_Prediction/dataset_tar_factor - 복사본.csv', parse_dates=['date'])
data.set_index('date', inplace=True)

# 월별 데이터에서 일별 데이터로 인덱스를 변경
date_range = pd.date_range(start=data.index.min(), end=data.index.max(), freq='D')
data_daily = data.reindex(date_range)

# 선형 보간 수행
data_daily.interpolate(method='linear', inplace=True)

# 결과 저장
data_daily.to_csv('C:/Users/Yoon Ha Hyeon/Desktop/CCI_Prediction/interpolated_daily_data.csv')
# %%
