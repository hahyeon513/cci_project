#%%
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error,mean_absolute_percentage_error
import matplotlib.pyplot as plt
import json


# 파일 경로 설정
file_paths = [
    r'C:\Users\Yoon Ha Hyeon\Desktop\CCI_Prediction\04_FAB_CCI\models\4_SVR\forecast_results_all_iterations_6month_tar1.csv',
    r'C:\Users\Yoon Ha Hyeon\Desktop\CCI_Prediction\04_FAB_CCI\models\4_SVR\forecast_results_all_iterations_6month_tar2.csv',
    r'C:\Users\Yoon Ha Hyeon\Desktop\CCI_Prediction\04_FAB_CCI\models\4_SVR\forecast_results_all_iterations_6month_tar3.csv',
    r'C:\Users\Yoon Ha Hyeon\Desktop\CCI_Prediction\04_FAB_CCI\models\4_SVR\forecast_results_all_iterations_6month_tar4.csv'
]

# 데이터 프레임 리스트 생성
dfs = []

# 각 파일에서 'Predicted' 열만 추출
for i, file_path in enumerate(file_paths):
    df = pd.read_csv(file_path)
    df = df[['Predicted']]  # 'Predicted' 열만 선택
    df.rename(columns={'Predicted': f'Predicted_Tar{i+1}'}, inplace=True)  # 열 이름 변경
    dfs.append(df)

# 모든 데이터 프레임을 수평으로 결합
df = pd.concat(dfs, axis=1)

# 결과 확인
print(df.head())

# %%
weights = [0.264, 0.096, 0.355, 0.285]
# 새로운 열 'Weighted_Sum'에 가중치 합 계산
df['Weighted_Sum1_4'] = df['Predicted_Tar1'] * weights[0] + df['Predicted_Tar2'] * weights[1] + df['Predicted_Tar3'] * weights[2] + df['Predicted_Tar4'] * weights[3]
df['Weighted_Sum1_4']  = df['Weighted_Sum1_4']*0.52
print(df.head())
# %%
# 파일 경로
file_path = r'C:\Users\Yoon Ha Hyeon\Desktop\CCI_Prediction\04_FAB_CCI\models\4_SVR\forecast_results_all_iterations_6month_tar5.csv'

# 파일에서 'Predicted' 열만 읽어오기
tar5_df = pd.read_csv(file_path)
tar5_df = tar5_df[['Predicted']]  # 'Predicted' 열 선택

# 기존 데이터 프레임 df에 'Predicted' 열 추가
df['Predicted_Tar5'] = tar5_df['Predicted']

# 'Predicted_Tar5' 열의 값을 0.48로 곱한 새로운 열 추가
df['Weighted_Tar5'] = df['Predicted_Tar5'] * 0.48

# 결과 확인
print(df.head())
# %%
df['Weighted_tar6']=df['Weighted_Sum1_4']+df['Weighted_Tar5']
print(df.head())

# %%
ile_path = r'C:\Users\Yoon Ha Hyeon\Desktop\CCI_Prediction\04_FAB_CCI\models\4_SVR\forecast_results_all_iterations_6month_tar6.csv'

# 파일에서 'Predicted' 열만 읽어오기
tar6_df = pd.read_csv(file_path)
tar6_df = tar6_df[['Predicted']]
df['Predicted_Tar6'] = tar6_df['Predicted']
print(df.head())
#%%
def calculate_mape(actual, forecast):
    actual, forecast = np.array(actual), np.array(forecast)
    non_zero_actual = actual != 0  # 실제 값이 0이 아닌 인덱스만 처리
    mape = np.mean(np.abs((actual[non_zero_actual] - forecast[non_zero_actual]) / actual[non_zero_actual])) * 100
    return mape
# MAPE 값 계산
mape_value = calculate_mape(df['Predicted_Tar6'], df['Weighted_tar6'])

print(f"The MAPE between 'Predicted_Tar6' and 'Weighted_tar6' is {mape_value:.2f}%")
# %%
# Save results
results_file_path = f'C:/Users/Yoon Ha Hyeon/Desktop/CCI_Prediction/mape_6month_weighted_vs_predicted.csv'
df.to_csv(results_file_path, index=False)
print(f"All results saved to {results_file_path}")
# %%
