#%%
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error,mean_absolute_percentage_error
import matplotlib.pyplot as plt
import json
from datetime import timedelta
from dateutil.relativedelta import relativedelta
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
from matplotlib.patches import Patch

class DataHandle:
    def __init__(self, target_data_path, factor_data_path, significant_vars_dict):
        self.target_df = pd.read_csv(str(target_data_path), index_col='date', parse_dates=True)
        self.factor_df = pd.read_csv(str(factor_data_path), index_col='date', parse_dates=True)
        self.df = self.target_df.join(self.factor_df, how='inner')
        self.significant_vars_dict = significant_vars_dict

    def set_data(self, target_code, k):
        filtered_target = self.target_df[target_code]
        filtered_factor = self.factor_df[self.significant_vars_dict[target_code][:k]]
        data = filtered_factor.join(filtered_target, how='inner')
        return data
    
    def cut_data(self, data, step, i):
        if i == 0:
            return data
        else:
            data_new = data.iloc[:-step*i]
            return data_new
    
    def differencing_to_stationarity(self, df, max_diff=3, signif=0.05):
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
                    series = series.diff().dropna()
                    diff_count += 1

            if diff_count > max_diff:
                print(f"Column '{column}' did not become stationary even after {max_diff} differencing.")
                diff_order[column] = np.nan

        return stationarized_df.bfill(), diff_order

class DataHandleSVR(DataHandle):  # Inherit from your existing DataHandle class
    
    def create_features(self, data, target_code, n_lags, additional_vars=[]):
        # Generate lagged features for the target and additional variables
        variables = additional_vars
        for var in variables:
            for lag in range(1, n_lags + 1):
                data[f'{var}_lag_{lag}'] = data[var].shift(lag)
        data = data.dropna()  # Drop rows with NaN values created by lagging
        return data   
    
    def find_best_lag(self, data_original, target_code, additional_vars):
        best_lag = None
        best_mape = float('inf')  # 가장 낮은 MAPE를 찾기 위한 초기값 설정
        max_lag = 12  # 최대 lag 값 설정
        offset_month = 6  # 교차 검증을 위해 데이터를 분할할 기간
        metrics = []

        # 각 lag 값에 대해 예측 성능을 평가
        for lag in range(1, max_lag + 1):
            # 현재 lag 값에 따라 데이터셋에 lagged features 생성
            temp_data = self.create_features(data_original, target_code, lag, additional_vars)
            
            # 데이터셋을 train 데이터와 test 데이터로 분할
            train_data = temp_data.iloc[:-offset_month, :]
            test_data = temp_data.iloc[-offset_month:, :]
            
            # SVR 모델을 학습시키고 예측 수행
            predictions = self.train_predict_svr(train_data, test_data, target_code, additional_vars)
            
            # MAPE 계산
            mse = mean_squared_error(test_data[target_code], predictions)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(test_data[target_code], predictions)
            mape = mean_absolute_percentage_error(test_data[target_code], predictions)
            
            metrics.append({
                'lag': lag,
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'mape': mape
            })
            
            # 현재 lag 값의 MAPE가 더 낮으면 최적의 lag 값 업데이트
            if mape < best_mape:
                best_mape = mape
                best_lag = lag

        metrics_df = pd.DataFrame(metrics)
        #metrics_df.to_csv('lag_metrics.csv', index=False)
        print(f"Optimized lag for features: {best_lag} with MAPE: {best_mape}")
        return best_lag, metrics_df


    def train_predict_svr(self, train_data, test_data, target_code, additional_vars):
        # Split features and target
        X_train = train_data.drop(columns=additional_vars)
        y_train = train_data[target_code]
        X_test = test_data.drop(columns=additional_vars)

        # Feature scaling for SVR
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # select optimized model hyperparameter and train the SVR model
        model = SVR(kernel='rbf', C=1000, gamma=0.001)
        model.fit(X_train_scaled, y_train)

        # Forecast
        predictions = model.predict(X_test_scaled)
        mape = mean_absolute_percentage_error(test_data[target_code], predictions)
        return predictions, mape
       
def load_dict_from_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

#%%
# Load json file
significant_vars_dict = load_dict_from_json('C:/Users/Yoon Ha Hyeon/Desktop/CCI_Prediction (3)/04_FAB_CCI/models/0_GCT/significant_vars_dict')

# Set parameters
target_code = 'tar6'
offset_month = 6
k = 3
 
# Load and handle data
DH = DataHandleSVR('C:/Users/Yoon Ha Hyeon/Desktop/CCI_Prediction (3)/04_FAB_CCI/dataset/dataset_target_cleaned.csv',
                   'C:/Users/Yoon Ha Hyeon/Desktop/CCI_Prediction (3)/04_FAB_CCI/dataset/dataset_factors_cleaned.csv',
                   significant_vars_dict)
all_results = pd.DataFrame()

# Set data
data_original = DH.set_data(target_code, k=k)
vars = data_original.columns
# Find best lag
best_lag, metrics_df = DH.find_best_lag(data_original, target_code, vars)

# Save metrics to CSV
metrics_file_path = f'C:/Users/Yoon Ha Hyeon/Desktop/CCI_Prediction (3)/OPTIMIZATION/lag/lag_metrics/lag_metrics_{target_code}.csv'
metrics_df.to_csv(metrics_file_path, index=False)
print(f"Metrics for each lag saved to {metrics_file_path}")

# Forecast using SVR
for i in range(60 // offset_month):
    data_cut = DH.cut_data(data=data_original, step=offset_month, i=i)
    optimized_data = DH.create_features(data_cut, target_code, best_lag, vars)
    
    train_data = optimized_data.iloc[:-offset_month,:]
    test_data = optimized_data.iloc[-offset_month:,:]
    
    # Predict using SVR
    predictions = DH.train_predict_svr(train_data, test_data, target_code, additional_vars=vars)
    forecast = pd.DataFrame(predictions, index=test_data.index, columns=[target_code])

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(data_cut.index, data_cut[target_code], label='Actual', color='blue')
    plt.plot(forecast.index, forecast[target_code], label='Predicted', color='red', linestyle='--')
    plt.xlabel('Date')
    plt.ylabel(target_code)
    plt.title(f'SVR Forecast for {target_code}')
    plt.legend()
    plt.show()

    # Collect results
    iteration_results = pd.DataFrame({
        'Date': forecast.index,
        'Iteration': i,
        'Actual': test_data[target_code],
        'Predicted': forecast[target_code]
    })
    all_results = pd.concat([all_results, iteration_results], ignore_index=True)

# Sort results by date
all_results.sort_values('Date', inplace=True)

# Save results
results_file_path = f'C:/Users/Yoon Ha Hyeon/Desktop/CCI_Prediction (3)/OPTIMIZATION/lag/lag_svr/forecast_results_all_iterations_lag_op_{offset_month}month_{target_code}.csv'
all_results.to_csv(results_file_path, index=False)
print(f"All results saved to {results_file_path}")
# %%