#%%
import pandas as pd

# Load raw dataset
df = pd.read_csv('./dataset/rawdata_factors.csv', parse_dates=['date'], index_col='date')

# Remove columns with any missing data
df_cleaned = df.dropna(axis=1, how='any')

# Set base date
base_date = pd.Timestamp('2015-01-01')

# Rescale the dataset with the base date (Base date=100)
base_values = df_cleaned.loc[base_date]
scaling_factors = 100 / base_values
scaled_df = df_cleaned * scaling_factors

# Save the scaled dataset to a new CSV file
cleaned_file_path = './dataset/dataset_factors.csv'
scaled_df.to_csv(cleaned_file_path)
print("done")
# %%
