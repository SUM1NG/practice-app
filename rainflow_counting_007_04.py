# 필요한 라이브러리 임포트
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from filterpy.kalman import KalmanFilter

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

    # 빈 값은 앞뒤 값의 평균으로 대체
    for y_name in y_names:
        df[y_name].interpolate(method='linear', limit_direction ='both', inplace=True)

    raw_data_dict = {x_name: df[x_name].tolist()}
    for y_name in y_names:
        raw_data_dict[y_name] = df[y_name].tolist()

    return df, raw_data_dict, x_units, y_units

"""
# 데이터 그래프 그리는 함수
def plot_data(df, x_name, y_names, x_units, y_units):
    # 그래프를 세로로 쌓음
    fig, axs = plt.subplots(len(y_names), 1, figsize=(20, 8))

    for ax, y_name, y_unit in zip(axs, y_names, y_units):
        ax.plot(df[x_name], df[y_name])
        ax.set_title(y_name)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel(y_unit)

        # 그래프에 세로선과 가로선 추가
        ax.grid(True)

        # 레이블을 왼쪽에 위치시킴
        ax.yaxis.tick_left()

    plt.tight_layout()
    plt.show()
"""
# 데이터 그래프 그리는 함수
def plot_both_data(df, kdf, x_name, y_names, x_units, y_units):
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

# 칼만 필터 적용하는 함수
def kalman_filter(df, y_names):
    for y_name in y_names:
        # 데이터를 담을 딕셔너리 생성
        filtered_data_dict = {x_name: df[x_name].tolist()}

        for y_name in y_names:
            measurements = np.array(df[y_name])

            # 칼만 필터 초기화
            kf = KalmanFilter(dim_x=1, dim_z=1)
            kf.F = np.array([[1.]])  # state transition matrix
            kf.H = np.array([[1.]])  # measurement function
            kf.x = np.array([0.])  # initial state
            kf.P *= 500.  # covariance matrix
            kf.R = 150  # state uncertainty
            kf.Q = 0.5  # process uncertainty

            # 칼만 필터 적용
            smoothed_state_means = np.zeros(measurements.shape)
            for i in range(len(measurements)):
                kf.predict()
                kf.update(measurements[i])
                smoothed_state_means[i] = kf.x

            df[y_name] = smoothed_state_means.flatten()

            # 필터링된 데이터를 딕셔너리에 저장
            filtered_data_dict[y_name] = df[y_name].tolist()

    return df, filtered_data_dict

def main():
    # 사용자로부터 필요한 정보 입력받기
    file_path = input("Enter the file path: ").replace('"', '')
    x_name = input("Enter the x-axis name: ").strip() # x_name에서 공백 제거
    y_names = [name.strip() for name in input("Enter the y-axis names (separated by comma): ").split(',')] # y_names에서 공백 제거

    # 데이터 처리
    df, x_units, y_units = process_data(file_path, x_name, y_names)

    # 새로운 CSV 파일 생성
    new_file_path = os.path.join(os.path.dirname(file_path), f"[processed]{os.path.basename(file_path)}")
    df.to_csv(new_file_path, index=False)

    # Raw Data 그래프 그리기
    #plot_data(df, x_name, y_names, x_units, y_units)

    # Kalman Filter 적용
    kdf = kalman_filter(df, y_names)

    # Filtered Data 그래프 그리기
    #plot_data(kdf, x_name, y_names, x_units, y_units)

    # Raw & Filtered Data 그래프 그리기
    plot_both_data(df, kdf, x_name, y_names, x_units, y_units)

if __name__ == "__main__":
    main()
