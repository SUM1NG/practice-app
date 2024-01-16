# 필요한 라이브러리 임포트
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from filterpy.kalman import KalmanFilter

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

# 데이터 처리하는 함수
def process_data(file_path, x_name, y_names):
    # 데이터 읽기
    df = pd.read_csv(file_path)

    # 불필요한 열 삭제
    df = df[[x_name] + y_names]

    # 공백 제거
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

    # 열 순서 재배치
    df = df[[x_name] + y_names]

    # 단위 저장 후 해당 행 삭제
    x_units = df.loc[1, x_name]
    y_units = [df.loc[1, y_name] for y_name in y_names]
    df = df.drop([0, 1])
    df = df.reset_index(drop=True)

    # 공백 제거
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

    # HH:MM:SS.FFF를 밀리초로 변환
    df[x_name] = df[x_name].apply(time_str_to_milliseconds)

    # 데이터 타입 확인 및 변환
    df = df.astype(float, errors='raise')
    
    # 동일한 값이 2번 이상 연속으로 나오는 경우 최소값으로 대체
    for y_name in y_names:
       df[y_name] = df[y_name].mask(df[y_name].eq(df[y_name].shift()) & df[y_name].eq(df[y_name].shift(-1)) & df[y_name].ne(0), df[y_name].min())

    # 동일한 `x_name` 값을 가진 행 제거
    df = df.drop_duplicates(subset=[x_name], keep='first')

    # 빈 값은 앞뒤 값의 평균으로 대체
    for y_name in y_names:
        df[y_name].interpolate(method='linear', limit_direction ='both', inplace=True)

    return df, x_units, y_units

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
def apply_kalman(df, x_name, y_names):
    # 원본 데이터프레임 복사
    kdf = df.copy()

    # 칼만 필터 적용
    for y_name in y_names:
        measurements = np.array(df[y_name])

        # 칼만 필터 초기화
        kf = KalmanFilter(dim_x=1, dim_z=1)
        kf.F = np.array([[1.]])  # state transition matrix
        kf.H = np.array([[1.]])  # measurement function
        kf.x = np.array([0.])  # initial state
        kf.P *= 100.  # covariance matrix
        kf.R = 5  # state uncertainty
        kf.Q = 2  # process uncertainty

        # 칼만 필터 적용
        smoothed_state_means = np.zeros(measurements.shape)
        for i in range(len(measurements)):
            kf.predict()
            kf.update(measurements[i])
            smoothed_state_means[i] = kf.x

        kdf[y_name] = smoothed_state_means.flatten()

    # NaN 값을 포함하는 행 제거
    kdf.dropna(inplace=True)

    return kdf

def calculate_filtering_percentage(original_df, filtered_df, y_names):
    percent = {}
    for y_name in y_names:
        original_data = original_df[y_name]
        filtered_data = filtered_df[y_name]
        deviation = np.mean(np.abs(original_data - filtered_data))
        percentage = (deviation / np.mean(np.abs(original_data))) * 100
        percent[y_name] = percentage
    return percent

# 데이터 그래프 그리는 함수
def plot_graphs(df, kdf, x_name, y_names, x_units, y_units):
    # 그래프를 세로로 쌓되, 각 y_axis에 대해 두 개의 그래프 (원시 데이터와 필터링된 데이터)를 가로로 배치.
    fig, axs = plt.subplots(len(y_names), 2, figsize=(25, 12))

    for i, y_name in enumerate(y_names):
        y_unit = y_units[i]
        
        # 원시 데이터 그래프
        axs[i, 0].plot(df[x_name], df[y_name], label='Raw Data')
        axs[i, 0].set_title(f'{y_name} (Raw Data)')
        axs[i, 0].set_xlabel('Time (ms)')
        axs[i, 0].set_ylabel(y_unit)
        axs[i, 0].grid(True)
        axs[i, 0].yaxis.tick_left()

        # 필터링된 데이터 그래프
        axs[i, 1].plot(kdf[x_name], kdf[y_name], label='Filtered Data', color='orange')
        axs[i, 1].set_title(f'{y_name} (Filtered Data)')
        axs[i, 1].set_xlabel('Time (ms)')
        axs[i, 1].set_ylabel(y_unit)
        axs[i, 1].grid(True)
        axs[i, 1].yaxis.tick_left()

    plt.tight_layout()
    plt.show()

def main():
    # 유저 입력 받기
    #file_path, x_name, y_names = get_user_input()
    file_path = "C:/Users/APTech-dev03/Documents/2. 프로젝트/KAI 제공자료/FA-50_VADR/test2.csv"
    x_name = "TIME"
    y_names = ["NZ","LONGACCEL","NY"]

    # 데이터 전처리
    df, x_units, y_units = process_data(file_path, x_name, y_names)

    # Raw Data 출력
    print("Saving raw data dictionary as csv:")
    output_to_csv(df, file_path, 'processed')


    # MAF Data 출력
    print("Saving MAF data dictionary as csv:")
    mdf = apply_maf(df, y_names, 1)
    mdf_percent = calculate_filtering_percentage(df, mdf, y_names)  # MAF에 대한 필터링 백분율
    for y_name in y_names:
        print(f"For {y_name}, the data was filtered by approximately {mdf_percent[y_name]:.2f}% in MAF")
    output_to_csv(mdf, file_path, 'maf')

    # Kalman Data 출력
    print("Saving Kalman data dictionary as csv:")
    kdf = apply_kalman(mdf, x_name, y_names)
    kdf_percent = calculate_filtering_percentage(df, kdf, y_names)  # Kalman 필터에 대한 필터링 백분율
    for y_name in y_names:
        print(f"For {y_name}, the data was filtered by approximately {kdf_percent[y_name]:.2f}% in Kalman filter")
    output_to_csv(kdf, file_path, 'kalman')

    # 그래프 출력
    plot_graphs(df, kdf, x_name, y_names, x_units, y_units)

# 메인 함수 콜
if __name__ == "__main__":
    main()



