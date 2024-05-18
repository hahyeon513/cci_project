#%%
import pandas as pd
import numpy as np
import json
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

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

def load_dict_from_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# Load json file
significant_vars_dict = load_dict_from_json('/home/chan/04_FAB_CCI/models/0_GCT/significant_vars_dict')

# Set parameters
target_code = 'tar6'
offset_month = 3
k = 1
lag_fit = 13

# Handle data
DH = DataHandle('../../dataset/dataset_target_cleaned.csv',
                '../../dataset/dataset_factors_cleaned.csv',
                significant_vars_dict)

# Filter data based on target code and factors
# Prepare DataFrame for all results
all_results = pd.DataFrame()

# Filter data based on target code and factors
for i in range(60//offset_month):
    data_original = DH.set_data(target_code, k=k)
    data_original = DH.cut_data(data=data_original, step=offset_month, i=i)

    # Differencing and PCA transformations
    data, diff_order = DH.differencing_to_stationarity(data_original)

    # Split dataset
    train_data = data.iloc[:-offset_month,:]
    test_data = data.iloc[-offset_month:,:]

    # Fit model
    try:
        model = VAR(train_data)
        fitted_model = model.fit(maxlags=15, ic='aic')  # Attempt to use AIC for optimal lag
    except Exception as e:
        print("Error fitting model with AIC: ", e)
        print("Falling back to a preset lag number.")
        fitted_model = model.fit(lag_fit)

    print(f"Selected Lag Order: {fitted_model.k_ar}")

    # Forecast with fitted model
    lag_order = fitted_model.k_ar
    laged_train_data = train_data.values[-lag_order:]
    forecast = pd.DataFrame(fitted_model.forecast(y=laged_train_data, steps=offset_month), index=test_data.index, columns=train_data.columns)

    forecast[target_code] = data_original[target_code].iloc[-offset_month-1] + forecast[target_code].cumsum()

    # Plot
    start_plot_index = data_original.index.get_loc(forecast.index[0]) - 36
    plt.figure(figsize=(10, 6))
    plt.plot(data_original[target_code].index[start_plot_index:], data_original[target_code].iloc[start_plot_index:], label='Actual', color='blue')
    plt.plot(forecast.index, forecast[target_code], label='Predicted', color='red', linestyle='--', marker='x')
    plt.xlabel('Date')
    plt.ylabel(target_code)
    plt.title(f'Actual vs Forecast for {target_code} (VAR)')
    plt.legend()
    plt.show()

    # Add results to the DataFrame
    iteration_results = pd.DataFrame({
        'Date': forecast.index,
        'Iteration': i,
        'Actual': data_original[target_code].iloc[-offset_month:],
        'Predicted': forecast[target_code]
    })
    all_results = pd.concat([all_results, iteration_results], ignore_index=True)

# Sort results by date
all_results.sort_values('Date', inplace=True)

# Save all results to a single CSV file
results_file_path = f'/home/chan/04_FAB_CCI/models/2_VAR/forecast_results_all_iterations_{offset_month}month_{target_code}.csv'
all_results.to_csv(results_file_path, index=False)
print(f"All results saved to {results_file_path}")
#%%
