#%%
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests
import json

class GrangerCausalityTest:
    def __init__(self, target_data_path, factor_data_path, factor_table_path, k, signif):
        self.target_df = pd.read_csv(str(target_data_path), index_col='date', parse_dates=True)
        self.factor_df = pd.read_csv(str(factor_data_path), index_col='date', parse_dates=True)
        self.df = self.target_df.join(self.factor_df, how='inner')
        self.df_code = pd.read_csv(factor_table_path, encoding='cp949')
        self.code_factor_dict = pd.Series(self.df_code.factor.values,index=self.df_code.code).to_dict()
        self.k = k
        self.signif = signif
        self.p_values_dict = {}
        self.significant_vars_dict = {}

    def cal_p_val(self, target_code, max_lag):
        if target_code not in self.p_values_dict:
            self.p_values_dict[target_code] = {}
        
        for column in self.factor_df.columns:
            test_result = grangercausalitytests(self.df[[column, target_code]], maxlag=max_lag, verbose=False)
            p_values = [test_result[i+1][0]['ssr_chi2test'][1] for i in range(max_lag)]
            min_p_value = np.min(p_values)
            self.p_values_dict[target_code][column] = min_p_value
    
    def get_top_k_factors(self, target_code):
        if target_code not in self.p_values_dict:
            return []
        significant_factors = [(var, p_val) for var, p_val in self.p_values_dict[target_code].items() if p_val < self.signif]
        significant_factors_sorted = sorted(significant_factors, key=lambda item: item[1])
        top_k_factors = significant_factors_sorted[:self.k] if self.k is not None else significant_factors_sorted
        return [var for var, _ in top_k_factors]
    
    def save_significant_vars_dict(self):
        with open('./significant_vars_dict_10', 'w') as file:
            json.dump(self.significant_vars_dict, file, indent=4)

    def save_significant_vars_dict_kor(self):
        dict_temp = {}
        for target in self.significant_vars_dict.keys():
            target_kor = self.code_factor_dict[target]
            vars_temp = []
            for var in self.significant_vars_dict[target]:
                vars_temp.append(self.code_factor_dict[var])
            dict_temp[target_kor] = vars_temp
        print(dict_temp)
        with open('./significant_vars_dict_kor_10', 'w') as file:
            json.dump(dict_temp, file,ensure_ascii = False, indent=4)

GCT = GrangerCausalityTest('../../dataset/dataset_target_cleaned.csv', 
                            '../../dataset/dataset_factors_cleaned_reset.csv',
                            '../../dataset/factor_table.csv',
                            k=15,
                            signif=0.05)

for i in range(len(GCT.target_df.columns)):
    target_code = 'tar' + str(i+1)
    GCT.cal_p_val(target_code, max_lag=14)
    top_factors = GCT.get_top_k_factors(target_code)
    GCT.significant_vars_dict[target_code] = top_factors

GCT.save_significant_vars_dict()
GCT.save_significant_vars_dict_kor()
#%%