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
from matplotlib.patches import Patch
from sklearn.model_selection import ParameterGrid

class DataHandle:
    def __init__(self, target_data_path, factor_data_path, significant_vars_dict):
        self.target_df = pd.read_csv(str(target_data_path), index_col = 'date', parse_dates=True)
        self.factor_df = pd.read_csv(str(factor_data_path), index_col = 'date', parse_dates=True)
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
   
    def train_predict_svr(self, train_data, test_data, target_code, additional_vars, params):
        X_train = train_data.drop(columns=additional_vars)
        y_train = train_data[target_code]
        X_test = test_data.drop(columns=additional_vars)
        
        # Feature scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train and predict with provided SVR parameters
        model = SVR(**params)
        model.fit(X_train_scaled, y_train)
        predictions = model.predict(X_test_scaled)
        
        mape = mean_absolute_percentage_error(test_data[target_code], predictions)
        return predictions, mape

    
    def SVR_param_selection(self, X, y, nfolds, target_code, k, offset_month):
        svr_parameters = {
            'kernel': ['rbf'],
            'gamma': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1],
            'C': [0.01, 0.1, 1, 10, 100, 1000]
        }
        
        param_grid = list(ParameterGrid(svr_parameters))
        best_params = None
        best_mape_avg = float('inf')
        all_mape_avg = []

        for params in param_grid:
            mape_values = []

            # Cross-validation loop
            for i in range(60 // offset_month):
                data_original = self.set_data(target_code, k=k)
                additional_vars = data_original.columns.tolist()  # Convert Index object to list
                data_original = self.cut_data(data=data_original, step=offset_month, i=i)
                data_original = self.create_features(data_original, target_code, n_lags=12, additional_vars=additional_vars)
                
                # Split dataset
                train_data = data_original.iloc[:-offset_month, :]
                test_data = data_original.iloc[-offset_month:, :]
                
                # Train and predict using SVR
                _, mape = self.train_predict_svr(train_data, test_data, target_code, additional_vars, params)
                mape_values.append(mape)

            mape_avg = np.mean(mape_values)
            all_mape_avg.append((params, mape_avg))

            if mape_avg < best_mape_avg:
                best_mape_avg = mape_avg
                best_params = params

        print(f"Best parameters found: {best_params}")
        return best_params, all_mape_avg


      
def load_dict_from_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data
#%%
# Load json file
significant_vars_dict = load_dict_from_json('C:/Users/Yoon Ha Hyeon/Desktop/CCI_Prediction (3)/04_FAB_CCI/models/0_GCT/significant_vars_dict')

# Set parameters
target_code = 'tar6'
offset_month = 3
nfolds = 10
k = 3

# Load and handle data
DH = DataHandleSVR('C:/Users/Yoon Ha Hyeon/Desktop/CCI_Prediction (3)/04_FAB_CCI/dataset/dataset_target_cleaned.csv',
                   'C:/Users/Yoon Ha Hyeon/Desktop/CCI_Prediction (3)/04_FAB_CCI/dataset/dataset_factors_cleaned.csv',
                   significant_vars_dict)
all_results = pd.DataFrame()

# Get the full dataset for hyperparameter tuning
data_full = DH.set_data(target_code, k=k)
data_full = DH.create_features(data_full, target_code, n_lags=12, additional_vars=data_full.columns)
X_full = data_full.drop(columns=[target_code])
y_full = data_full[target_code]

# Feature scaling for the full dataset
scaler = StandardScaler()
X_full_scaled = scaler.fit_transform(X_full)

# Perform hyperparameter tuning on the full dataset
best_params, all_mape_avg = DH.SVR_param_selection(X_full_scaled, y_full.values.ravel(), nfolds, target_code, k, offset_month)

# Print the best parameters
print(f"Best parameters: {best_params}")
for params, mape_avg in all_mape_avg:
    print(f"Params: {params}, Average MAPE: {mape_avg}")

# Forecast using SVR
mape_values = []    
for i in range(60 // offset_month):
    data_original = DH.set_data(target_code, k=k)
    vars = data_original.columns.tolist()  # Convert Index object to list
    data_original = DH.cut_data(data=data_original, step=offset_month, i=i)
    data_original = DH.create_features(data_original, target_code, n_lags=12, additional_vars=vars)
    
    # Split dataset
    train_data = data_original.iloc[:-offset_month, :]
    test_data = data_original.iloc[-offset_month:, :]
    
    # Predict using SVR
    predictions, mape = DH.train_predict_svr(train_data, test_data, target_code, additional_vars=vars, params=best_params)
    mape_values.append(mape)
    forecast = pd.Series(predictions, index=test_data.index)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(data_original.index, data_original[target_code], label='Actual', color='blue')
    plt.plot(forecast.index, forecast.values, label='Predicted', color='red', linestyle='--')
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
        'Predicted': forecast.values
    })
    all_results = pd.concat([all_results, iteration_results], ignore_index=True)

# Sort results by date
all_results.sort_values('Date', inplace=True)

# Save results
results_file_path = f'C:/Users/Yoon Ha Hyeon/Desktop/CCI_Prediction (3)/OPTIMIZATION/hyper/hyper_svr/svr_hyper_op_forecast_results_all_iterations_{offset_month}month_{target_code}.csv'
all_results.to_csv(results_file_path, index=False)
print(f"All results saved to {results_file_path}")

# Print the average MAPE
average_mape = np.mean(mape_values)
print(f"Average MAPE over all iterations: {average_mape}")
#%%