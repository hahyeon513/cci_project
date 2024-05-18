#%%
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from math import sqrt
import pmdarima as pm
from sklearn.model_selection import TimeSeriesSplit

# Set target code for analysis
'''
1 : 골조
2 : 건축마감
3 : 설비마감
4 : 전기마감
5 : 통합
'''
target_num = '5'
target_code = 'tar' + target_num

# Set model parameters
model_params = {'ADF_signif': 0.05,
                'test_month': 3}

# Load dataset
df_tar = pd.read_csv('./dataset/dataset_target.csv', index_col='date', parse_dates=True)
df_tar = df_tar[[target_code]]
df_tar = df_tar.asfreq('MS')

# Conduct ADF Test to check stationarity
def check_stationarity(timeseries, signif=model_params['ADF_signif']):
    from statsmodels.tsa.stattools import adfuller
    adf_test_result = adfuller(timeseries, autolag='AIC')
    return adf_test_result[1] <= signif

# Make the series stationary if needed
if not check_stationarity(df_tar):
    print(f'data diffed')
    df_tar_staionarized = df_tar.diff().bfill()
    
total_data_points = len(df_tar)  # 데이터 포인트의 총 개수
test_set_size = model_params['test_month'] # 테스트 세트 크기: 3개월
min_train_length = 60  # 최소 5년 (최소 훈련 데이터 길이)
n_splits = (total_data_points - test_set_size - min_train_length) // test_set_size # 가능한 분할 수 계산

print("n_splits:",n_splits)
#%%
# 교차 검증 시 데이터 분할 시각화 그래프

cmap_data = plt.cm.Paired
cmap_cv = plt.cm.coolwarm
plt.style.use('fivethirtyeight')

def plot_cv_indices(cv, X, n_splits, lw=10):
    
    fig, ax = plt.subplots()
    """Create a sample plot for indices of a cross-validation object."""

    # Generate the training/testing visualizations for each CV split
    for ii, (tr, tt) in enumerate(cv.split(X=X)):
        # Fill in indices with the training/test groups
        indices = np.array([np.nan] * len(X))
        indices[tt] = 1
        indices[tr] = 0

        # Visualize the results
        ax.scatter(range(len(indices)), [ii + .5] * len(indices),
                   c=indices, marker='_', lw=lw, cmap=cmap_cv,
                   vmin=-.2, vmax=1.2)

    # Formatting
    yticklabels = list(range(n_splits))
    ax.set(yticks=np.arange(n_splits) + .5, yticklabels=yticklabels,
           xlabel='Sample index', ylabel="CV iteration",
           ylim=[n_splits+0.1, -.1], xlim=[0, len(X)])
    ax.set_title('{}'.format(type(cv).__name__), fontsize=15)
    
    ax.legend([Patch(color=cmap_cv(.8)), Patch(color=cmap_cv(.02))],
          ['Testing set', 'Training set'], loc=(1.02, .8))

XX = np.arange(len(df_tar))
tscv = TimeSeriesSplit(n_splits=n_splits)
plot_cv_indices(tscv, XX, n_splits=n_splits)

#%%
print(df_tar)
print(len(df_tar))
#%%
# 훈련 및 테스트 분할을 설정하는 사용자 정의 함수
def custom_time_series_split(data, initial_train_months, test_months, step_months):
    total_points = len(data)
    train_months_to_points = initial_train_months
    test_months_to_points = test_months
    step_months_to_points = step_months
    
    indices = np.arange(total_points)
    
    while train_months_to_points + test_months_to_points <= total_points:
        train_indices = indices[:train_months_to_points]
        test_indices = indices[train_months_to_points:train_months_to_points + test_months_to_points]
        
        yield data.iloc[train_indices], data.iloc[test_indices]
        
        # Update the next training end point by adding step months to the current training data
        train_months_to_points += step_months_to_points

# 교차 검증 설정
initial_train_months = 60  # 초기 훈련 기간 (5년)
test_months = 3           # 테스트 기간 (3개월)
step_months = 3            # 추가 훈련 기간 (3개월)

# DataFrame 생성하여 분할 인덱스 저장
all_splits = []
cv_splits = custom_time_series_split(df_tar, initial_train_months, test_months, step_months)

for i, (train_data, test_data) in enumerate(cv_splits):
    all_splits.append({
        'split': i + 1,
        'train_start': train_data.index.min(),
        'train_end': train_data.index.max(),
        'test_start': test_data.index.min(),
        'test_end': test_data.index.max()
    })

df_splits = pd.DataFrame(all_splits)

# DataFrame 출력
print(df_splits)

# DataFrame을 CSV 파일로 저장 (경로는 사용자의 환경에 맞게 조정)
df_splits.to_csv('time_series_cv_splits.csv', index=False)

# %%

# Conduct ADF Test to check stationarity
def check_stationarity(timeseries, signif=model_params['ADF_signif']):
    from statsmodels.tsa.stattools import adfuller
    adf_test_result = adfuller(timeseries, autolag='AIC')
    return adf_test_result[1] <= signif

# Make the series stationary if needed
def make_stationary(data):
    if not check_stationarity(data):
        return data.diff().dropna()  # 차분을 통해 정상성 확보
    return data

# 교차 검증 분할을 위한 함수
def perform_arima_analysis(train_data, test_data):
    train_data = make_stationary(train_data)
    test_data = make_stationary(test_data)  # 테스트 데이터도 정상 시계열로 변환

    # ARIMA 모델 최적화 및 피팅
    model = pm.auto_arima(train_data, seasonal=False, stepwise=True, trace=False, suppress_warnings=True)
    model_fit = model.fit(train_data)

    # 예측
    forecast = model_fit.predict(n_periods=len(test_data))
    forecast = pd.Series(forecast, index=test_data.index)  # 예측 결과를 시리즈로 변환

    # 평가 지표 계산
    mae = mean_absolute_error(test_data, forecast)
    mse = mean_squared_error(test_data, forecast)
    rmse = sqrt(mse)
    mape = np.mean(np.abs((test_data - forecast) / test_data)) * 100

    return mae, mse, rmse, mape
#%%
# 결과를 저장할 리스트
results = []

# df_splits에 정의된 각 시간 범위에 대해 반복
for index, row in df_splits.iterrows():
    train_start, train_end = row['train_start'], row['train_end']
    test_start, test_end = row['test_start'], row['test_end']

    # 훈련 및 테스트 데이터 세트 분할
    train_data = df_tar.loc[train_start:train_end]
    test_data = df_tar.loc[test_start:test_end]

    # ARIMA 분석 수행 및 결과 저장
    mae, mse, rmse, mape = perform_arima_analysis(train_data[target_code], test_data[target_code])
    results.append({
        'Split': index + 1,
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape
    })

# 결과를 DataFrame으로 변환하고 출력
df_results = pd.DataFrame(results)
print(df_results)
# %%
