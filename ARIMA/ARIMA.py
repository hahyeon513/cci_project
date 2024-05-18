#%%
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from math import sqrt
import pmdarima as pm

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

# Split the dataset into train/test dataset
test_month = model_params['test_month']
train_data = df_tar_staionarized.iloc[:-test_month,:]
test_data = df_tar_staionarized.iloc[-test_month:,:]

# Optimize the parameters(p,d,q) for the model with auto-arima
'''
returns (p,d,q)
p = 
d = 
q = 
'''
model_auto = pm.auto_arima(train_data, seasonal=False, stepwise=True, trace=True, error_action='ignore', suppress_warnings=True)
best_order = model_auto.order
print(f"Best ARIMA Order: {best_order}")

# Fit the ARIMA model
model = ARIMA(train_data, order=best_order)
model_fit = model.fit()

# Forecast the target for test months
forecast = model_fit.forecast(steps=test_month)
forecast = df_tar[target_code].iloc[-test_month-1] + forecast.cumsum()

# Plot figure: set starting point(base month) for the plot
plt.figure(figsize=(10, 6))
two_years_before_forecast_start = test_data.index[0] - pd.DateOffset(years=2)
plot_start_date = max(two_years_before_forecast_start, df_tar.index[0])
plt.plot(df_tar.loc[plot_start_date:].index, df_tar.loc[plot_start_date:], marker='o', label='Actual (Including 2 Years Before Forecast)')

# Plot details
plt.plot(df_tar.index[-test_month:], forecast, marker='x', linestyle='--', label='Forecasted')
plt.title(f'ARIMA Forecast vs Actuals for {target_code}(ARIMA)')
plt.xlabel('Date')
plt.ylabel('Values')
plt.legend()
plt.savefig(f'./result/{target_num}/actual_vs_forecast_tar{target_num}_testmonthof{test_month}.png')
plt.show()

# Extract actual values for the test period
actual = df_tar[target_code].iloc[-test_month:]

# Calculate performance metrics
mae = mean_absolute_error(actual, forecast)
mse = mean_squared_error(actual, forecast)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((actual - forecast) / actual)) * 100

# Print the metrics
print(f'Mean Absolute Error (MAE): {mae:.3f}')
print(f'Mean Squared Error (MSE): {mse:.3f}')
print(f'Root Mean Squared Error (RMSE): {rmse:.3f}')
print(f'Mean Absolute Percentage Error (MAPE): {mape:.2f}%')
#%%