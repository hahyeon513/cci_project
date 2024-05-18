#%%
import pandas as pd
from sklearn import preprocessing

# Load raw dataset
df = pd.read_csv('C:/Users/Yoon Ha Hyeon/Documents/CCI_Prediction/VAR/dataset/rawdata_factors.csv', parse_dates=['date'], index_col='date')

# Remove columns with any missing data
df_cleaned = df.dropna(axis=1, how='any')

# Set base date
base_date = pd.Timestamp('2015-01-01')

# Rescale the dataset with the base date (Base date=100)
base_values = df_cleaned.loc[base_date]
scaling_factors = 100 / base_values
scaled_df = df_cleaned * scaling_factors

# Save the scaled dataset to a new CSV file
cleaned_file_path = 'C:/Users/Yoon Ha Hyeon/Documents/CCI_Prediction/VAR/dataset/dataset_factors.csv'
scaled_df.to_csv(cleaned_file_path)
print("done")
print("original 변수 개수: ", len(df.columns))
print("preprocess 변수 개수: ",len(scaled_df.columns))
scaled_df
# %%
'''
# normalization 할시
from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
scaled_df_normalized_data = scaler.fit_transform(scaled_df)
scaled_df_normalized_df = pd.DataFrame(scaled_df_normalized_data, index=scaled_df.index, columns=scaled_df.columns)
scaled_df_normalized_df

print(len(scaled_df_normalized_df.columns))
cleaned_normalized_path = '../dataset/dataset_factor_noramlized.csv'
scaled_df_normalized_df.to_csv(cleaned_normalized_path)
'''
# %%
