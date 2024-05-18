#%%
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error,mean_absolute_percentage_error
import matplotlib.pyplot as plt
import json

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
    
    def train_predict_svr(self, train_data, test_data, target_code, additional_vars):
        # Split features and target
        X_train = train_data.drop(columns=additional_vars)
        y_train = train_data[target_code]
        X_test = test_data.drop(columns=additional_vars)

        # Feature scaling for SVR
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Initialize and train the SVR model
        model = SVR(kernel='rbf', C=1000, gamma=0.001, epsilon=0.1)
        model.fit(X_train_scaled, y_train)

        # Forecast
        predictions = model.predict(X_test_scaled)
        return predictions

def load_dict_from_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# Load json file
significant_vars_dict = load_dict_from_json('/Users/hahyeon/Desktop/CCI_Prediction (3)/04_FAB_CCI/models/0_GCT/significant_vars_dict')

# Set parameters
target_code = 'tar2'
offset_month = 6
k = 3

# Load and handle data
DH = DataHandleSVR('/Users/hahyeon/Desktop/CCI_Prediction (3)/04_FAB_CCI/dataset/dataset_target_cleaned.csv',
                   '/Users/hahyeon/Desktop/CCI_Prediction (3)/04_FAB_CCI/dataset/dataset_factors_cleaned.csv',
                   significant_vars_dict)
all_results = pd.DataFrame()

# Forecast using SVR
for i in range(60 // offset_month):
    data_original = DH.set_data(target_code, k=k)
    vars = data_original.columns
    data_original = DH.cut_data(data=data_original, step=offset_month, i=i)
    data_original = DH.create_features(data_original, target_code, n_lags=12, additional_vars=vars)

    # Split dataset
    train_data = data_original.iloc[:-offset_month,:]
    test_data = data_original.iloc[-offset_month:,:]

    # Predict using SVR
    predictions = DH.train_predict_svr(train_data, test_data, target_code, additional_vars=vars)
    forecast = pd.DataFrame(predictions, index=test_data.index, columns=[target_code])
    
    # Calculate MAPE
    mape = mean_absolute_percentage_error(test_data[target_code], forecast[target_code])
    print(f"Iteration {i} - MAPE: {mape}")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(data_original.index, data_original[target_code], label='Actual', color='blue')
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
results_file_path = f'/Users/hahyeon/Desktop/CCI_Prediction (3)/SVM/forecast_results_all_iterations_{offset_month}month_{target_code}.csv'
all_results.to_csv(results_file_path, index=False)
print(f"All results saved to {results_file_path}")
# %%
