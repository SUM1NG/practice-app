# 필요한 라이브러리 임포트
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from filterpy.kalman import KalmanFilter
from scipy.signal import butter, lfilter

# 사용자로부터 필요한 정보 입력받기
def get_user_input():
    file_path = input("Enter the file path: ").replace('"', '')
    x_name = input("Enter the x-axis name: ").strip()
    y_names = [name.strip() for name in input("Enter the y-axis names (separated by comma): ").split(',')]
    return file_path, x_name, y_names

# 결과를 CSV 파일로 출력하는 함수
def output_to_csv(df, file_path, prefix):
    new_file_path = os.path.join(os.path.dirname(file_path), f"[{prefix}]{os.path.basename(file_path)}")
    df.to_csv(new_file_path, index=False)
    
# 시간 문자열(HH:MM:SS.FFF)을 밀리초로 변환하는 함수
def time_str_to_milliseconds(time_str):
    # 예외 처리 추가
    try:
        h, m, rest = time_str.split(':')
        s, ms = map(float, rest.split('.'))
        milliseconds = ((int(h) * 60 + int(m)) * 60 + s) * 1000 + ms
    except ValueError:
        print("오류: 시간 문자열이 'HH:MM:SS.FFF' 형식에 맞지 않습니다.")
        return None
    return milliseconds

# 이상치(Outlier) 필터 함수
def outlier_filter(data, min_val, max_val):
    # 이상치 위치 찾기
    outlier_indices = [i for i, x in enumerate(data) if x < min_val or x > max_val]
    
    # 각 이상치에 대해
    for i in outlier_indices:
        # 이상치의 앞뒤 100개 값의 리스트 생성
        surrounding_data = data[max(0, i-100) : min(len(data), i+100+1)]
        
        # 이상치가 아닌 값들만 선택
        surrounding_data = [x for x in surrounding_data if min_val <= x <= max_val]
        
        # 이상치를 주변 값들의 평균으로 대체
        if surrounding_data:  # surrounding_data가 비어 있지 않은 경우에만
            data[i] = sum(surrounding_data) / len(surrounding_data)
        else:
            data[i] = 0  # surrounding_data가 비어 있는 경우 (모두 이상치인 경우), 0으로 대체

    return data

# MAF Moving Average Filter(이동 평균 필터) 적용하는 함수
def apply_maf(df, y_names, window_size):
    # 원본 데이터프레임 복사
    mdf = df.copy()

    # MAF 적용
    for y_name in y_names:
        mdf[y_name] = df[y_name].rolling(window=window_size, center=True).mean()
    
    # NaN 값을 포함하는 행 제거
    mdf.dropna(inplace=True)

    return mdf

# Kalman Filter(칼만 필터) 적용하는 함수
"""
칼만 필터는 시스템이 시간에 따라 어떻게 변화하는지, 
그리고 이 변화에 얼마나 확신할 수 있는지에 대한 불확실성을 수치화
이를 위해 공분산 행렬 P와 시스템 노이즈 행렬 Q 및 측정 노이즈 R을 사용

kf.P (공분산 행렬): 시스템 상태의 불확실성
초기화에서는 상태의 초기 불확실성을 나타냅니다. 
값이 크면 상태 추정치에 대한 불확실성이 크다는 것을 의미하며, 
값이 작으면 불확실성이 작다는 것을 의미합니다. 
이 값이 커지면 칼만 필터는 측정값을 더 신뢰하게 되고, 
작아지면 현재 상태 추정치를 더 신뢰하게 됩니다.

kf.R (측정 노이즈): 측정 장치의 노이즈
이 값이 크면 측정 장치가 많은 노이즈를 생성한다고 가정하며, 
이 경우 칼만 필터는 측정값보다 모델을 더 신뢰하게 됩니다. 
반대로 이 값이 작으면 측정 장치가 정확하다고 가정하며, 
이 경우 칼만 필터는 측정값을 더 신뢰하게 됩니다.

kf.Q (시스템 노이즈 행렬): 시스템 자체에서 발생할 수 있는 무작위 노이즈
이 값이 크면 시스템 모델이 많은 노이즈를 생성한다고 가정하며, 
이 경우 칼만 필터는 측정값을 더 신뢰하게 됩니다. 
반대로 이 값이 작으면 시스템 모델이 정확하다고 가정하며, 
이 경우 칼만 필터는 모델을 더 신뢰하게 됩니다.

이러한 값들을 조정함으로써 칼만 필터의 행동을 제어할 수 있습니다. 
그러나 특정 값을 너무 크게 하거나 작게 하면 필터의 성능에 부정적인 영향을 미칠 수 있습니다. 
예를 들어, 측정 노이즈 R이 너무 크면 필터는 측정값을 무시하게 될 수 있으며, 
시스템 노이즈 Q가 너무 작으면 필터는 모델을 과도하게 신뢰하게 될 수 있습니다. 
따라서 이러한 값들은 주의해서 조정해야 합니다.
"""

def apply_kalman(df, x_name, y_names, P_value, R_value, Q_value):
    # 원본 데이터프레임 복사
    kdf = df.copy()  # 원본 데이터프레임을 복사하여 새로운 데이터프레임 생성

    # 칼만 필터 적용
    for y_name in y_names:  # 각 y_name에 대해
        measurements = np.array(df[y_name])  # 측정값을 NumPy 배열로 변환

        kf = KalmanFilter(dim_x=1, dim_z=1)  # 칼만 필터 객체 생성
        kf.F = np.array([[1.]])  # 상태 전이 행렬 설정
        kf.H = np.array([[1.]])  # 측정 함수 설정
        kf.x = np.array([0.])  # 초기 상태 설정
        kf.P *= P_value  # 공분산 행렬 설정 (covariance matrix, 상태의 초기 불확실성)
        kf.R = R_value  # 측정 불확실성 설정 (state uncertainty)
        kf.Q = Q_value  # 과정 불확실성 설정 (process uncertainty)

        # 칼만 필터 적용
        smoothed_state_means = np.zeros(measurements.shape)  # 스무딩된 상태 평균을 저장할 배열 초기화
        for i in range(len(measurements)):  # 각 측정값에 대해
            kf.predict()  # 상태 예측
            kf.update(measurements[i])  # 상태 업데이트
            smoothed_state_means[i] = kf.x  # 스무딩된 상태 평균 저장

        kdf[y_name] = smoothed_state_means.flatten()  # 스무딩된 상태 평균을 데이터프레임에 저장

    # NaN 값을 포함하는 행 제거
    kdf.dropna(inplace=True)  # NaN 값을 포함하는 행 제거

    return kdf  # 처리된 데이터프레임 반환

def calculate_filtering_percentage(original_df, filtered_df, y_names):
    percent = {}
    for y_name in y_names:
        original_data = original_df[y_name]
        filtered_data = filtered_df[y_name]
        deviation = np.mean(np.abs(original_data - filtered_data))
        percentage = (deviation / np.mean(np.abs(original_data))) * 100
        percent[y_name] = percentage
    return percent

# 데이터 처리하는 함수
def process_data(file_path, x_name, y_names, filter_min, filter_max):
    # 데이터 읽기
    df = pd.read_csv(file_path)

    # 불필요한 열 삭제
    df = df[[x_name] + y_names]

    # 공백 제거
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

    # 열 순서 재배치
    df = df[[x_name] + y_names]

    # 단위 저장 후 해당 행 삭제
    x_units = df.loc[0, x_name]
    y_units = [df.loc[0, y_name] for y_name in y_names]
    df = df.drop([0])
    df = df.reset_index(drop=True)

    # 공백 제거
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

    # HH:MM:SS.FFF를 밀리초로 변환
    df[x_name] = df[x_name].apply(time_str_to_milliseconds)

    # 데이터 타입 확인 및 변환
    df = df.astype(float, errors='raise')

    # 동일한 `x_name` 값을 가진 행 제거
    df = df.drop_duplicates(subset=[x_name], keep='first')

    # 빈 값은 앞뒤 값의 평균으로 대체
    for y_name in y_names:
        df[y_name].interpolate(method='linear', limit_direction ='both', inplace=True)

    # 이상치 필터 적용
    for y_name in y_names:
        df[y_name] = outlier_filter(list(df[y_name]), filter_min, filter_max)

    return df, x_units, y_units

# 데이터 그래프 그리는 함수
def plot_graphs(df, kdf, mdf, x_name, y_names, x_units, y_units, mdf_percent, kdf_percent):
    
    # 그래프를 세로로 쌓되, 각 y_axis에 대해 두 개의 그래프 (원시 데이터와 필터링된 데이터)를 가로로 배치.
    fig, axs = plt.subplots(len(y_names), 3, figsize=(25, 12))

    for i, y_name in enumerate(y_names):
        y_unit = y_units[i]
        
        # 원시 데이터 그래프
        axs[i, 0].plot(df[x_name], df[y_name], label='Raw Data')
        axs[i, 0].set_title(f'{y_name} (Raw Data)')
        axs[i, 0].set_xlabel('Time (ms)')
        axs[i, 0].set_ylabel(y_unit)
        axs[i, 0].grid(True)
        axs[i, 0].yaxis.tick_left()

        # 필터링된 데이터 그래프 (MAF)
        axs[i, 1].plot(mdf[x_name], mdf[y_name], label='Filtered Data (MAF)', color='orange')
        axs[i, 1].set_title(f'{y_name} (Filtered Data, MAF, filtered: {mdf_percent[y_name]:.2f}%)')
        axs[i, 1].set_xlabel('Time (ms)')
        axs[i, 1].set_ylabel(y_unit)
        axs[i, 1].grid(True)
        axs[i, 1].yaxis.tick_left()

        # 필터링된 데이터 그래프 (Kalman)
        axs[i, 2].plot(kdf[x_name], kdf[y_name], label='Filtered Data (Kalman)', color='red')
        axs[i, 2].set_title(f'{y_name} (Filtered Data, Kalman, filtered: {kdf_percent[y_name]:.2f}%)')
        axs[i, 2].set_xlabel('Time (ms)')
        axs[i, 2].set_ylabel(y_unit)
        axs[i, 2].grid(True)
        axs[i, 2].yaxis.tick_left()

    plt.tight_layout()
    plt.show()

def main():
    # 유저 입력 받기
    #file_path, x_name, y_names = get_user_input()

    # Input File (Temporary, for testing)
    file_path = "C:/Users/APTech-dev03/Documents/2. 프로젝트/KAI 제공자료/FA-50_VADR/test2.csv"
    x_name = "TIME"
    y_names = ["NZ","LONGACCEL","NY"]

    # Outlier Filter Config
    filter_min = -100
    filter_max = 100

    # MAF Config
    window_size = 2

    # Kalman Filter Config
    P_value = 1000
    R_value = 5
    Q_value = 10

    # 데이터 전처리
    df, x_units, y_units = process_data(file_path, x_name, y_names, filter_min, filter_max)

    # Raw Data 출력
    print("Saving raw data dictionary as csv:")
    output_to_csv(df, file_path, 'processed')

    # MAF Data 출력
    print("Saving MAF data dictionary as csv:")
    mdf = apply_maf(df, y_names, window_size)
    mdf_percent = calculate_filtering_percentage(df, mdf, y_names)  # MAF에 대한 필터링 백분율
    for y_name in y_names:
        print(f"For {y_name}, the data was filtered by approximately {mdf_percent[y_name]:.2f}% in MAF")
    output_to_csv(mdf, file_path, 'maf')

    # Kalman Data 출력
    print("Saving Kalman data dictionary as csv:")
    kdf = apply_kalman(mdf, x_name, y_names, P_value, R_value, Q_value)
    kdf_percent = calculate_filtering_percentage(df, kdf, y_names)  # Kalman 필터에 대한 필터링 백분율
    for y_name in y_names:
        print(f"For {y_name}, the data was filtered by approximately {kdf_percent[y_name]:.2f}% in Kalman filter")
    output_to_csv(kdf, file_path, 'kalman')

    # 그래프 출력
    #plot_graphs(df, kdf, x_name, y_names, x_units, y_units)
    plot_graphs(df, kdf, mdf, x_name, y_names, x_units, y_units, mdf_percent, kdf_percent)

# 메인 함수 콜
if __name__ == "__main__":
    main()