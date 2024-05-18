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
        
    def calculate_mape(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        non_zero = y_true != 0
        return np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100    


class DataHandleSVR(DataHandle):  # Inherit from your existing DataHandle class
    def create_features(self, data, target_code, n_lags=12, additional_vars=[]):
        # Generate lagged features for the target and additional variables
        variables = additional_vars
        for var in variables:
            for lag in range(1, n_lags + 1):
                data[f'{var}_lag_{lag}'] = data[var].shift(lag)
        data = data.dropna()  # Drop rows with NaN values created by lagging
        return data
   
    def train_predict_svr(self, train_data, test_data, target_code, additional_vars, nfolds):
        X_train = train_data.drop(columns=additional_vars) 
        y_train = train_data[target_code]
        X_test = test_data.drop(columns=additional_vars)
        #y_test = test_data[target_code]
        
        #feature scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 그리드 서치를 통해 최적의 하이퍼파라미터를 찾습니다.
        best_params = self.SVR_param_selection(X_train_scaled, y_train.values.ravel(), nfolds)
        model = SVR(**best_params)           
        
        # 초기화, 모델 훈련
        model.fit(X_train_scaled, y_train)
        
        #예측
        predictions = model.predict(X_test_scaled)
        mape = mean_absolute_percentage_error(test_data[target_code], predictions)
        print(f"MAPE: {mape}")
        return predictions

    
    def SVR_param_selection(self, X, y, nfolds):
        svr_parameters = [
            {'kernel': ['rbf'],
            'gamma': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1],
            'C': [0.01, 0.1, 1, 10, 100, 1000]}
        
        ]
        # 그리드 서치 설정, 회귀에서는 'neg_mean_squared_error' 등의 회귀 관련 스코어를 사용
        clf = GridSearchCV(SVR(), svr_parameters, scoring='neg_mean_squared_error', cv=nfolds)
        clf.fit(X, y)
        print(clf.best_params_)
        # 최적 파라미터 반환
        return clf.best_params_

      
def load_dict_from_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data
#%%
# Load json file
significant_vars_dict = load_dict_from_json('C:/Users/Yoon Ha Hyeon/Desktop/CCI_Prediction/04_FAB_CCI/models/0_GCT/significant_vars_dict')

# Set parameters
target_code = 'tar6'
offset_month = 6
nfolds = 10
k = 5


# Load and handle data
DH = DataHandleSVR('C:/Users/Yoon Ha Hyeon/Desktop/CCI_Prediction/04_FAB_CCI/dataset/dataset_target_cleaned.csv',
                   'C:/Users/Yoon Ha Hyeon/Desktop/CCI_Prediction/04_FAB_CCI/dataset/dataset_factors_cleaned.csv',
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
    predictions = DH.train_predict_svr(train_data, test_data, target_code, additional_vars=vars, nfolds=nfolds)
    forecast = pd.DataFrame(predictions, index=test_data.index, columns=[target_code])
    
    
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
results_file_path = f'C:/Users/Yoon Ha Hyeon/Desktop/CCI_Prediction/SVM/forecast_results_all_iterations_{offset_month}month_{target_code}.csv'
all_results.to_csv(results_file_path, index=False)
print(f"All results saved to {results_file_path}")