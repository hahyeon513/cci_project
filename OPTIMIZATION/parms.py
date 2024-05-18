#%%
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import json
from datetime import timedelta
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
from matplotlib.patches import Patch
from sklearn.metrics import mean_absolute_percentage_error
from itertools import product

class DataHandlemodel:
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

    def create_features(self, data, target_code, n_lags, additional_vars=[]):
        # Generate lagged features for the target and additional variables
        variables = additional_vars
        for var in variables:
            for lag in range(1, n_lags + 1):
                data[f'{var}_lag_{lag}'] = data[var].shift(lag)
        data = data.dropna()  # Drop rows with NaN values created by lagging
        return data

    def load_dict_from_json(file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
    
class DataHandleSVR(DataHandlemodel):
    def SVR_param_selection(self, X, y, nfolds, svr_params):
        # Use provided SVR parameters
        clf = SVR(**svr_params)
        # Fit the model
        clf.fit(X, y)
        return clf

        
    def train_predict_svr_params(self, train_data, test_data, target_code, additional_vars, nfolds,svr_params):
            X_train = train_data.drop(columns = additional_vars) 
            y_train = train_data[target_code]
            X_test = test_data.drop(columns = additional_vars)
            y_test = test_data[target_code]
            
            #feature scaling
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            model = self.SVR_param_selection(X_train_scaled, y_train.values.ravel(), nfolds, svr_params)

            # Predict
            predictions = model.predict(X_test_scaled)
            return predictions
#%%
# Set parameters
target_code = 'tar6'
model = 'svr'
offset_month = 6
nfolds = 10
k_max = 10
lag_max =12

# Generate all combinations of SVR parameters, k, and lag values
kernel_options = ['rbf']
gamma_options = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
C_options = [0.01, 0.1, 1, 10, 100, 1000]
k_options = list(range(1, k_max+1))  
lag_options = list(range(1, lag_max+1))  

param_combinations = list(product(kernel_options, gamma_options, C_options, k_options, lag_options))

best_mape = float('inf')  # Initialize the best MAPE value
best_params = None  # Initialize the best parameter combination
best_forecast = None  # Initialize the best forecast
best_test_data = None  # Initialize the test data associated with the best forecast

# Loop over each combination
for kernel, gamma, C, k, lag in param_combinations:
    svr_params = {'kernel': kernel, 'gamma': gamma, 'C': C}  
    
    # Load json file
    significant_vars_dict = DataHandleSVR.load_dict_from_json('C:/Users/Yoon Ha Hyeon/Desktop/CCI_Prediction/04_FAB_CCI/models/0_GCT/significant_vars_dict')
    DH = DataHandleSVR('C:/Users/Yoon Ha Hyeon/Desktop/CCI_Prediction/04_FAB_CCI/dataset/dataset_target_cleaned.csv',
                   'C:/Users/Yoon Ha Hyeon/Desktop/CCI_Prediction/04_FAB_CCI/dataset/dataset_factors_cleaned.csv',
                   significant_vars_dict)
    all_results = pd.DataFrame()

    # Forecast using SVR
    for i in range(60 // offset_month):
        data_original = DH.set_data(target_code, k=k)
        vars = data_original.columns
        data_original = DH.cut_data(data=data_original, step=offset_month, i=i)
        data_original = DH.create_features(data_original, target_code, n_lags=lag, additional_vars=vars)

        # Split dataset
        train_data = data_original.iloc[:-offset_month,:]
        test_data = data_original.iloc[-offset_month:,:]
        
        # Predict using SVR with grid search for SVR hyperparameters
        predictions = DH.train_predict_svr_params(train_data, test_data, target_code, additional_vars=vars, nfolds=nfolds, svr_params=svr_params)
        forecast = pd.DataFrame(predictions, index=test_data.index, columns=[target_code])
        
        # Calculate MAPE
        mape = mean_absolute_percentage_error(test_data[target_code], forecast[target_code])
        
        # Update best parameters if MAPE is lower
        if mape < best_mape:
            best_mape = mape
            best_params = {'kernel': kernel, 'gamma': gamma, 'C': C, 'k': k, 'lag': lag}
            best_forecast = forecast
            best_test_data = test_data
        # Collect results
        iteration_results = pd.DataFrame({
            'Date': forecast.index,
            'Actual': test_data[target_code],
            'Predicted': forecast[target_code]
        })
        all_results = pd.concat([all_results, iteration_results], ignore_index=True)

    # Sort results by date
    all_results.sort_values('Date', inplace=True)

# Save results
    
# Create a more file-friendly parameter string
param_str = "_".join([f"{key}{value}" for key, value in best_params.items()]).replace('{', '').replace('}', '').replace(',', '_').replace(' ', '')

# Now use this in the filename
results_file_path = f'C:/Users/Yoon Ha Hyeon/Desktop/CCI_Prediction/OPTIMIZATION/forecast_results_all_iterations_{model}_{offset_month}month_{target_code}_{param_str}.csv'
all_results.to_csv(results_file_path, index=False)
print(f"Best result saved to {results_file_path}")    
         
# Now best_params contains the best combination of hyperparameters for SVR, k, and lag
print("Best parameters:", best_params)

# Plot results for the best forecast
plt.figure(figsize=(10, 6))
plt.plot(data_original.index, data_original[target_code], label='Actual', color='blue')
plt.plot(forecast.index, forecast[target_code], label='Predicted', color='red', linestyle='--')
plt.xlabel('Date')
plt.ylabel(target_code)
plt.title(f'SVR Forecast for {target_code}')
plt.legend()
plt.show()
# %%
