#%%
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt
import json
from datetime import timedelta

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

class DataHandleXGB(DataHandle):  # Inherit from your existing DataHandle class
    def create_features(self, data, target_code, n_lags, additional_vars=[]):
        # Generate lagged features for the target and additional variables
        variables = additional_vars
        for var in variables:
            for lag in range(1, n_lags + 1):
                data[f'{var}_lag_{lag}'] = data[var].shift(lag)
        data = data.dropna()  # Drop rows with NaN values created by lagging
        return data   
    
    def train_predict_xgboost(self, train_data, test_data, target_code, additional_vars):
        X_train = train_data.drop(columns=additional_vars)
        y_train = train_data[target_code]
        X_test = test_data.drop(columns=additional_vars)
        
        # Train and predict with XGBoost
        model = XGBRegressor(colsample_bytree=1.0, learning_rate=0.2, max_depth= 7, n_estimators=200, subsample= 1.0)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        mape = mean_absolute_percentage_error(test_data[target_code], predictions)
        return predictions, mape

    def find_best_lag(self, target_code, k, offset_month):
        best_lag = None
        best_mape_avg = float('inf')
        max_lag = 12
        all_mape_avg = []

        for lag in range(1, max_lag + 1):
            mape_values = []

            # Cross-validation loop
            for i in range(60 // offset_month):
                data_original = self.set_data(target_code, k=k)
                additional_vars = data_original.columns.tolist()
                data_original = self.cut_data(data=data_original, step=offset_month, i=i)
                data_original = self.create_features(data_original, target_code, n_lags=lag, additional_vars=additional_vars)
                
                # Split dataset
                train_data = data_original.iloc[:-offset_month, :]
                test_data = data_original.iloc[-offset_month:, :]
                
                # Train and predict using XGBoost
                _, mape = self.train_predict_xgboost(train_data, test_data, target_code, additional_vars)
                mape_values.append(mape)

            mape_avg = np.mean(mape_values)
            all_mape_avg.append((lag, mape_avg))

            if mape_avg < best_mape_avg:
                best_mape_avg = mape_avg
                best_lag = lag

        print(f"Best lag found: {best_lag}")
        return best_lag, all_mape_avg

def load_dict_from_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# Load json file
significant_vars_dict = load_dict_from_json('C:/Users/Yoon Ha Hyeon/Desktop/CCI_Prediction (3)/04_FAB_CCI/models/0_GCT/significant_vars_dict')

# Set parameters
target_code = 'tar5'
offset_month = 6
k = 3

# Load and handle data
DH = DataHandleXGB('C:/Users/Yoon Ha Hyeon/Desktop/CCI_Prediction (3)/04_FAB_CCI/dataset/dataset_target_cleaned.csv',
                   'C:/Users/Yoon Ha Hyeon/Desktop/CCI_Prediction (3)/04_FAB_CCI/dataset/dataset_factors_cleaned.csv',
                   significant_vars_dict)
all_results = pd.DataFrame()

# Find the best lag
best_lag, all_mape_avg = DH.find_best_lag(target_code, k, offset_month)

# Print the best lag and the average MAPE for each lag
print(f"Best lag: {best_lag}")
for lag, mape_avg in all_mape_avg:
    print(f"Lag: {lag}, Average MAPE: {mape_avg}")

# Forecast using XGBoost
mape_values = []    
for i in range(60 // offset_month):
    data_original = DH.set_data(target_code, k=k)
    vars = data_original.columns.tolist()
    data_original = DH.cut_data(data=data_original, step=offset_month, i=i)
    data_original = DH.create_features(data_original, target_code, n_lags=best_lag, additional_vars=vars)
    
    # Split dataset
    train_data = data_original.iloc[:-offset_month, :]
    test_data = data_original.iloc[-offset_month:, :]
    
    # Predict using XGBoost
    predictions, mape = DH.train_predict_xgboost(train_data, test_data, target_code, additional_vars=vars)
    mape_values.append(mape)
    forecast = pd.Series(predictions, index=test_data.index)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(data_original.index, data_original[target_code], label='Actual', color='blue')
    plt.plot(forecast.index, forecast.values, label='Predicted', color='red', linestyle='--')
    plt.xlabel('Date')
    plt.ylabel(target_code)
    plt.title(f'XGBoost Forecast for {target_code}')
    plt.legend()
    plt.show()

    # Collect results
    iteration_results = pd.DataFrame({
        'Date': forecast.index,
        'Iteration': i,
        'Actual': test_data[target_code],
        'Predicted': forecast.values
    })
    all_results = pd.concat([all_results, iteration_results], ignore_index=True)

# Sort results by date
all_results.sort_values('Date', inplace=True)

# Save results
results_file_path = f'C:/Users/Yoon Ha Hyeon/Desktop/CCI_Prediction (3)/OPTIMIZATION/lag/lag_xgboost/forecast_results_all_iterations_lag_op_{offset_month}month_{target_code}.csv'
all_results.to_csv(results_file_path, index=False)
print(f"All results saved to {results_file_path}")

# Print the average MAPE
average_mape = np.mean(mape_values)
print(f"Average MAPE over all iterations: {average_mape}")
#%%