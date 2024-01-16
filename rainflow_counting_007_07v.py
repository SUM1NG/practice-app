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

# Rainflow Counting 함수
def apply_rainflow(df, y_names):
    rf_cycles = {}  # 결과를 저장할 딕셔너리

    # 각 y_name에 대해 Rainflow Counting 적용
    for y_name in y_names:
        rdf = df[y_name]  # 해당 y_name의 데이터
        stack = []
        cycles = []

        for point in rdf:
            while len(stack) >= 3:
                # 4-point condition
                if min(stack[-2], stack[-1]) < point <= max(stack[-2], stack[-1]):
                    break
                else:
                    # found a cycle, pop the last two points from stack
                    cycle = (stack.pop(), stack.pop())
                    cycles.append(cycle)

            stack.append(point)

        # handling the residual stack
        while len(stack) > 2:
            cycle = (stack.pop(), stack.pop())
            cycles.append(cycle)

        rf_cycles[y_name] = cycles  # 각 y_name별로 계산된 cycles를 결과 딕셔너리에 저장

    return rf_cycles  # 결과 딕셔너리 반환

# 도수분포표 그리는 함수
def draw_frequency_distribution(result_dict):
    fig, axs = plt.subplots(1, len(result_dict), figsize=(6*len(result_dict), 10))
    for ax, (y_name, cycles) in zip(axs, result_dict.items()):
        # 빈도 분포 계산
        frequencies = [cycle[1] - cycle[0] for cycle in cycles]

        # 그래프 그리기
        ax.hist(frequencies, bins=100, density=True)
        ax.set_title(f"Frequency Distribution for {y_name}")
        ax.set_xlabel("Cycle")
        ax.set_ylabel("Frequency")
    plt.tight_layout()
    plt.show()

# 3D 히스토그램 그리는 함수
def draw_3d_histogram(result_dict):
    fig = plt.figure(figsize=(6*len(result_dict), 10))
    for i, (y_name, cycles) in enumerate(result_dict.items(), 1):
        # 데이터 준비
        x_data = [cycle[0] for cycle in cycles]
        y_data = [cycle[1] for cycle in cycles]
        hist, x_edges, y_edges = np.histogram2d(x_data, y_data, bins=20)

        # 3D 그래프 그리기
        ax = fig.add_subplot(1, len(result_dict), i, projection='3d')
        x_pos, y_pos = np.meshgrid(x_edges[:-1] + 0.25, y_edges[:-1] + 0.25)
        x_pos = x_pos.flatten('F')
        y_pos = y_pos.flatten('F')
        z_pos = np.zeros_like(x_pos)
        dx = dy = 0.5 * np.ones_like(z_pos)
        dz = hist.flatten()

        ax.bar3d(x_pos, y_pos, z_pos, dx, dy, dz, color='b', zsort='average')
        ax.set_title(f"3D Histogram for {y_name}")
        ax.set_xlabel('Cycle Start')
        ax.set_ylabel('Cycle End')
        ax.set_zlabel('Count')
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

    # Rainflow Counting 출력
    print("Saving Rainflow data dictionary as csv:")
    cycles = apply_rainflow(kdf, y_names)
    output_to_csv(kdf, file_path, 'rainflow')
  
    # 그래프 출력
    plot_graphs(df, kdf, mdf, x_name, y_names, x_units, y_units, mdf_percent, kdf_percent)

    # Rainflow Counting 결과 출력
    draw_frequency_distribution(cycles)  # 도수분포표 출력
    draw_3d_histogram(cycles)  # 히스토그램 출력

# 메인 함수 콜
if __name__ == "__main__":
    main()