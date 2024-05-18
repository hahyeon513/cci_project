#%%
import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# Set target code
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
                'max_lag': 10,
                'Granger_signif': 0.02,
                'test_month': 6}

# Load dataset for targets
df_tar = pd.read_csv('./dataset/dataset_target.csv', index_col='date', parse_dates=True)
df_tar = df_tar[[target_code]]
df_tar = df_tar.asfreq('MS')

# Load dataset for leading factors
df_var = pd.read_csv('./dataset/dataset_factors.csv', index_col='date', parse_dates=True)
df_var = df_var.asfreq('MS')

# Join dataframe(target+factors)
df = df_tar.join(df_var, how='inner')

# Differencing until the stationarity is confirmed for all column
def differencing_to_stationarity(df, max_diff=3, signif=model_params['ADF_signif']):
    stationarized_df = df.copy()
    diff_order = {}

    for column in df.columns:
        series = df[column]
        diff_count = 0
        
        while diff_count <= max_diff:
            adf_test_result = adfuller(series, autolag='AIC')
            p_value = adf_test_result[1]

            if p_value <= signif:
                print(f"Column '{column}' became stationary after {diff_count} differencing.")
                diff_order[column] = diff_count
                stationarized_df[column] = series
                break
            else:
                # Conduct differencing if the series is not stationary
                series = series.diff().dropna()
                diff_count += 1

        if diff_count > max_diff:
            print(f"Column '{column}' did not become stationary even after {max_diff} differencing.")
            diff_order[column] = np.nan

    return stationarized_df, diff_order

# Conduct differencing
stationarized_df, diff_orders = differencing_to_stationarity(df)

# Save differencing results for all variables
diff_orders_df = pd.DataFrame.from_dict(diff_orders, orient='index', columns=['Differencing_Order'])
diff_orders_df.to_csv(f'./result/{target_num}/diff_orders_target_{target_num}.csv')

# Preprocess Null values for all columns
stationarized_df = stationarized_df.bfill()

# Conduct Granger causaulity test to select optimal variables for prediction
max_lag = model_params['max_lag']
granger_causality_results = {}

for column in stationarized_df.columns:
    if column != target_code:
        test_result = grangercausalitytests(stationarized_df[[target_code, column]], maxlag=max_lag, verbose=False)
        min_p_value = np.min([test_result[lag][0]['ssr_chi2test'][1] for lag in range(1, max_lag + 1)])
        granger_causality_results[column] = min_p_value

# Select optimal variables for prediction
signif_p_val = model_params['Granger_signif']
significant_vars_granger = [target_code] + [var for var, p_val in granger_causality_results.items() if p_val <= signif_p_val]
significant_data_granger = stationarized_df[significant_vars_granger]

print(f'\n*****{len(significant_vars_granger)} variables were chosen from total {len(df.columns)}*****\n')

# Save GCT results to csv
significant_data_granger.to_csv(f'./result/{target_num}/significant_vars_{target_num}.csv')

# Split dataset
test_month = model_params['test_month']
train_data = significant_data_granger.iloc[:-test_month,:]
test_data = significant_data_granger.iloc[-test_month:,:]


from statsmodels.tsa.api import VAR

# 가정: train_data는 이미 정의되어 있고, 적절히 전처리된 상태입니다.

model = VAR(train_data)
results_aic = []

for p in range(1, 20):
    try:
        results = model.fit(p)
        results_aic.append(results.aic)
    except ValueError as e:
        print(f"라그 {p}에서 문제 발생: {e}")
        results_aic.append(float('inf')) # 오류가 발생한 경우, AIC 값을 무한대로 설정하여 해당 라그를 사용하지 않도록 함

# AIC 값이 가장 낮은 라그 선택
optimal_p = results_aic.index(min(results_aic)) + 1  # 인덱스는 0부터 시작하므로 1을 더함
print(f"선택된 최적의 라그 수: {optimal_p}")

# 최적의 라그로 모델 재피팅
best_model = model.fit(optimal_p)


#%%
# Plot AIC results
plt.plot(list(np.arange(1,20,1)), results_aic)
plt.xlabel("Order")
plt.ylabel("AIC")
plt.show()

# Declare optimal lag based on the AIC result
optimal_lag = np.argmin(results_aic) + 1
print(f'Optimal lag: {optimal_lag}')
model_fitted = model.fit(optimal_lag)

# Conduct prediction
forecast_df = pd.DataFrame(model_fitted.forecast(y=train_data.values[-optimal_lag:], 
                                                 steps=test_month), 
                                                 index=test_data.index, 
                                                 columns=test_data.columns)

forecast_df[target_code] = df[target_code].iloc[-test_month-1] + forecast_df[target_code].cumsum()

# Plot actual vs prediction results
start_plot_index = df_tar.index.get_loc(forecast_df.index[0]) - 36
plt.figure(figsize=(10, 6))
plt.plot(df_tar.index[start_plot_index:], df_tar[target_code].iloc[start_plot_index:], label='Actual', color='blue')
plt.plot(forecast_df.index, forecast_df[target_code], label='Forecast', color='red')
plt.xlabel('Date')
plt.ylabel(target_code)
plt.title(f'Actual vs Forecast for {target_code} (VAR with {len(significant_vars_granger)})')
plt.legend()
plt.savefig(f'./result/{target_num}/actual_vs_forecast_tar{target_num}_testmonthof{test_month}.png')
plt.show()

# Calculate performance metrics
mae = mean_absolute_error(df_tar[target_code].iloc[-test_month:], forecast_df[target_code])
mse = mean_squared_error(df_tar[target_code].iloc[-test_month:], forecast_df[target_code])
rmse = np.sqrt(mse)
mape = np.mean(np.abs((df_tar[target_code].iloc[-test_month:] - forecast_df[target_code]) / df_tar[target_code].iloc[-test_month:])) * 100

# Print performance
print(f'MAE: {mae}')
print(f'MSE: {mse}')
print(f'RMSE: {rmse}')
print(f'MAPE: {mape}%')

# Save performance results
performance_metrics = pd.DataFrame({'MAE': [mae], 'MSE': [mse], 'RMSE': [rmse], 'MAPE': [mape]})
performance_metrics.to_csv(f'./result/{target_num}/performance_metrics_{target_num}.csv', index=False)

# Save prediction results to csv
actual_vs_forecast = pd.concat([df_tar[target_code].iloc[-test_month:], forecast_df[target_code]], axis=1)
actual_vs_forecast.columns = ['Actual', 'Forecast']
actual_vs_forecast.to_csv(f'./result/{target_num}/actual_vs_forecast_data_{target_num}.csv')
#%%