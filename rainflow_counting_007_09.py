# 필요한 라이브러리 임포트
import os

# 필요한 패키지들이 설치되어 있는지 확인하고, 없다면 설치
try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import numpy as np
    import tkinter as tk
    from tkinter import font as tkFont, ttk
    from filterpy.kalman import KalmanFilter
except ImportError:
    os.system('pip install pandas matplotlib numpy tkinter filterpy')  # 필요한 패키지가 없으면 설치


# 사용자로부터 필요한 정보 입력받기
def get_user_input():
    file_path = input("Enter the file path: ").replace('"', '')  # 파일 경로 입력 받음
    x_name = input("Enter the x-axis name: ").strip()  # x축 이름 입력 받음
    y_names = [name.strip() for name in input("Enter the y-axis names (separated by comma): ").split(',')]  # y축 이름들 입력 받음
    return file_path, x_name, y_names  # 입력 받은 정보 반환


# 데이터 처리하는 함수
def process_data(file_path, x_name, y_names, filter_min, filter_max):
    # 데이터 읽기
    df = pd.read_csv(file_path)  # csv 파일 읽어와서 DataFrame 생성

    # 불필요한 열 삭제
    df = df[[x_name] + y_names]  # 필요한 열만 선택해서 DataFrame 재생성

    # 공백 제거
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)  # 각 열의 값이 문자열인 경우, 앞뒤 공백 제거

    # 열 순서 재배치
    df = df[[x_name] + y_names]  # 열 순서를 [x_name, y_names] 순서로 재배치

    # 단위 저장 후 해당 행 삭제
    x_units = df.loc[0, x_name]  # x축 단위 저장
    y_units = [df.loc[0, y_name] for y_name in y_names]  # y축 단위들 저장
    df = df.drop([0])  # 단위를 저장한 첫 번째 행 삭제
    df = df.reset_index(drop=True)  # 인덱스 재설정

    # 공백 제거
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)  # 각 열의 값이 문자열인 경우, 앞뒤 공백 제거

    # HH:MM:SS.FFF를 밀리초로 변환
    df[x_name] = df[x_name].apply(time_str_to_milliseconds)  # 시간 문자열을 밀리초로 변환하는 함수 적용

    # 데이터 타입 확인 및 변환
    df = df.astype(float, errors='raise')  # 각 열의 값들을 실수로 변환

    # 동일한 `x_name` 값을 가진 행 제거
    df = df.drop_duplicates(subset=[x_name], keep='first')  # x_name 값이 동일한 행 중 첫 번째 행만 남기고 삭제

    # 빈 값은 앞뒤 값의 평균으로 대체
    for y_name in y_names:  # 각 y_name에 대해
        df[y_name].interpolate(method='linear', limit_direction ='both', inplace=True)  # 빈 값이 있는 경우, 앞뒤 값의 선형적인 평균으로 대체

    # 이상치 필터 적용
    for y_name in y_names:  # 각 y_name에 대해
        df[y_name] = outlier_filter(list(df[y_name]), filter_min, filter_max)  # 이상치 필터 적용

    return df, x_units, y_units  # 처리된 DataFrame과 단위들 반환


# 결과를 CSV 파일로 출력하는 함수
def output_to_csv(df, file_path, prefix):
    new_file_path = os.path.join(os.path.dirname(file_path), f"[{prefix}]{os.path.basename(file_path)}")  # 새 파일 경로 생성
    df.to_csv(new_file_path, index=False)  # DataFrame을 csv 파일로 저장
    

# Rainflow Counting 결과를 CSV로 저장하는 함수
def output_rf_cycles_to_csv(rf_cycles, file_path, prefix):
    # 각 y_name에 대한 cycles를 별도의 열로 가지는 DataFrame 생성
    max_len = max(len(cycles) for cycles in rf_cycles.values())  # 가장 긴 cycles의 길이 계산
    data = {y_name: pd.Series(cycles, dtype='object') for y_name, cycles in rf_cycles.items()}  # 각 y_name에 대한 cycles를 Series로 변환하여 딕셔너리 생성
    df = pd.DataFrame(data)  # 딕셔너리를 이용하여 DataFrame 생성

    # DataFrame을 CSV로 저장
    new_file_path = os.path.join(os.path.dirname(file_path), f"[{prefix}]{os.path.basename(file_path)}")  # 새 파일 경로 생성
    df.to_csv(new_file_path, index=False)  # DataFrame을 csv 파일로 저장


# 시간 문자열(HH:MM:SS.FFF)을 밀리초로 변환하는 함수
def time_str_to_milliseconds(time_str):
    # 예외 처리 추가
    try:
        h, m, rest = time_str.split(':')  # 시간 문자열을 시, 분, 나머지로 분리
        s, ms = map(float, rest.split('.'))  # 나머지를 초, 밀리초로 분리
        milliseconds = ((int(h) * 60 + int(m)) * 60 + s) * 1000 + ms  # 밀리초로 변환
    except ValueError:  # 형식이 맞지 않는 경우 예외 처리
        print("오류: 시간 문자열이 'HH:MM:SS.FFF' 형식에 맞지 않습니다.")
        return None
    return milliseconds  # 밀리초 반환


# 이상치(Outlier) 필터 함수
def outlier_filter(data, min_val, max_val):
    # 이상치 위치 찾기
    outlier_indices = [i for i, x in enumerate(data) if x < min_val or x > max_val]  # 이상치의 인덱스를 찾음
    
    # 각 이상치에 대해
    for i in outlier_indices:
        # 이상치의 앞뒤 100개 값의 리스트 생성
        surrounding_data = data[max(0, i-100) : min(len(data), i+100+1)]  # 이상치 주변의 데이터를 얻음
        
        # 이상치가 아닌 값들만 선택
        surrounding_data = [x for x in surrounding_data if min_val <= x <= max_val]  # surrounding_data에서 이상치를 제외
        
        # 이상치를 주변 값들의 평균으로 대체
        if surrounding_data:  # surrounding_data가 비어 있지 않은 경우에만
            data[i] = sum(surrounding_data) / len(surrounding_data)  # 이상치를 surrounding_data의 평균으로 대체
        else:
            data[i] = 0  # surrounding_data가 비어 있는 경우 (모두 이상치인 경우), 0으로 대체

    return data  # 처리된 데이터 반환


# MAF Moving Average Filter(이동 평균 필터) 적용하는 함수
def apply_maf(df, y_names, window_size):
    # 원본 데이터프레임 복사
    mdf = df.copy()  # 원본 데이터프레임을 복사하여 새로운 데이터프레임 생성

    # MAF 적용
    for y_name in y_names:  # 각 y_name에 대해
        mdf[y_name] = df[y_name].rolling(window=window_size, center=True).mean()  # 이동 평균 필터 적용
    
    # NaN 값을 포함하는 행 제거
    mdf.dropna(inplace=True)  # NaN 값을 포함하는 행 제거

    return mdf  # 처리된 데이터프레임 반환


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
            smoothed_state_means[i] = kf.x[0]  # 스무딩된 상태 평균 저장

        kdf[y_name] = smoothed_state_means.flatten()  # 스무딩된 상태 평균을 데이터프레임에 저장

    # NaN 값을 포함하는 행 제거
    kdf.dropna(inplace=True)  # NaN 값을 포함하는 행 제거

    return kdf  # 처리된 데이터프레임 반환


# 각 필터로 인해 제거된 데이터량% 연산
def calculate_filtering_percentage(original_df, filtered_df, y_names):
    percent = {}  # 결과를 저장할 딕셔너리 생성
    for y_name in y_names:  # 각 y_name에 대해
        original_data = original_df[y_name]  # 원본 데이터 선택
        filtered_data = filtered_df[y_name]  # 필터링된 데이터 선택
        deviation = np.mean(np.abs(original_data - filtered_data))  # 원본 데이터와 필터링된 데이터의 차이의 평균을 계산
        percentage = (deviation / np.mean(np.abs(original_data))) * 100  # 편차를 원본 데이터의 평균으로 나눈 후 백분율로 변환
        percent[y_name] = percentage  # y_name별 편차 백분율을 딕셔너리에 저장

    return percent  # 결과 딕셔너리 반환


# MAF 및 Kalman 필터 결과 그래프 그리는 함수
def plot_graphs(df, kdf, mdf, x_name, y_names, x_units, y_units, mdf_percent, kdf_percent):
    # 그래프를 세로로 쌓되, 각 y_axis에 대해 두 개의 그래프 (원시 데이터와 필터링된 데이터)를 가로로 배치.
    fig, axs = plt.subplots(len(y_names), 3, figsize=(25, 12))  # 서브플롯 생성

    for i, y_name in enumerate(y_names):  # 각 y_name에 대해
        y_unit = y_units[i]  # y_unit 선택

        # 원시 데이터 그래프
        axs[i, 0].plot(df[x_name], df[y_name], label='Raw Data')  # 원시 데이터 그래프 그리기
        axs[i, 0].set_title(f'{y_name} (Raw Data)')  # 제목 설정
        axs[i, 0].set_xlabel('Time (ms)')  # x축 레이블 설정
        axs[i, 0].set_ylabel(y_unit)  # y축 레이블 설정
        axs[i, 0].grid(True)  # 그리드 표시
        axs[i, 0].yaxis.tick_left()  # y축 틱을 왼쪽에 표시

        # 필터링된 데이터 그래프 (MAF)
        axs[i, 1].plot(mdf[x_name], mdf[y_name], label='Filtered Data (MAF)', color='orange')  # 필터링된 데이터 그래프 그리기
        axs[i, 1].set_title(f'{y_name} (Filtered Data, MAF, filtered: {mdf_percent[y_name]:.2f}%)')  # 제목 설정
        axs[i, 1].set_xlabel('Time (ms)')  # x축 레이블 설정
        axs[i, 1].set_ylabel(y_unit)  # y축 레이블 설정
        axs[i, 1].grid(True)  # 그리드 표시
        axs[i, 1].yaxis.tick_left()  # y축 틱을 왼쪽에 표시

        # 필터링된 데이터 그래프 (Kalman)
        axs[i, 2].plot(kdf[x_name], kdf[y_name], label='Filtered Data (Kalman)', color='red')  # 필터링된 데이터 그래프 그리기
        axs[i, 2].set_title(f'{y_name} (Filtered Data, Kalman, filtered: {kdf_percent[y_name]:.2f}%)')  # 제목 설정
        axs[i, 2].set_xlabel('Time (ms)')  # x축 레이블 설정
        axs[i, 1].set_ylabel(y_unit)  # y축 레이블 설정
        axs[i, 1].grid(True)  # 그리드 표시
        axs[i, 1].yaxis.tick_left()  # y축 틱을 왼쪽에 표시

    plt.tight_layout()  # 레이아웃 조정
    plt.show()  # 그래프 보여주기

# Rainflow Counting 함수
def apply_rainflow(df, y_names):
    rf_cycles = {}  # 결과를 저장할 딕셔너리 생성

    # 각 y_name에 대해 Rainflow Counting 적용
    for y_name in y_names:
        rdf = df[y_name]  # 해당 y_name의 데이터 선택
        stack = []  # 스택 초기화
        cycles = []  # 싸이클 저장할 리스트 초기화

        for point in rdf:  # 각 데이터 포인트에 대해
            while len(stack) >= 3:  # 스택의 크기가 3 이상인 동안
                # 4-point condition 확인
                if min(stack[-2], stack[-1]) < point <= max(stack[-2], stack[-1]):
                    break
                else:
                    # cycle 발견, 스택에서 마지막 두 점 제거
                    cycle = (stack.pop(), stack.pop())
                    cycles.append(cycle)  # cycle을 리스트에 추가

            stack.append(point)  # 현재 포인트를 스택에 추가

        # residual stack 처리
        while len(stack) > 2:  # 스택의 크기가 2 초과인 동안
            cycle = (stack.pop(), stack.pop())  # 스택에서 마지막 두 점 제거 
            cycles.append(cycle)  # cycle을 리스트에 추가

        rf_cycles[y_name] = cycles  # 각 y_name별로 계산된 cycles를 결과 딕셔너리에 저장

    return rf_cycles  # 결과 딕셔너리 반환


# 히스토그램 출력 함수
def draw_graphs(rf_cycles):
    # 2D와 3D 히스토그램이 모두 들어갈 수 있도록 충분히 큰 figure 생성
    fig = plt.figure(figsize=(8 * len(rf_cycles), 8))  # figure 크기 설정

    # 색상 그라디언트 생성
    cmap = mcolors.LinearSegmentedColormap.from_list("", ["blue", "red"])  # 파랑에서 빨강으로 변하는 그라디언트

    # 2D 히스토그램 그리기
    for i, (y_name, cycles) in enumerate(rf_cycles.items(), 1):  # 각 싸이클에 대해
        ax = fig.add_subplot(2, len(rf_cycles), i)  # subplot 추가
        frequencies = [cycle[1] - cycle[0] for cycle in cycles]  # 싸이클의 주파수 계산
        n, bins, patches = ax.hist(frequencies, bins=100, density=True)  # 히스토그램 생성
        
        # 히스토그램 막대에 색상 그라디언트 적용
        bin_centers = 0.5 * (bins[:-1] + bins[1:])  # 각 bin의 중심값 계산
        col = bin_centers - min(bin_centers)  # 최소값을 빼서 0에서 시작하도록 함
        col /= max(col)  # 최대값으로 나눠서 0~1 사이로 정규화
        for c, p in zip(col, patches):  # 각 bin에 대해
            plt.setp(p, 'facecolor', cmap(c))  # 색상 설정

        ax.set_title(f"Frequency Distribution for {y_name}")  # 제목 설정
        ax.set_xlabel("Cycle")  # x축 레이블 설정
        ax.set_ylabel("Frequency")  # y축 레이블 설정

    # 3D 히스토그램 그리기
    for i, (y_name, cycles) in enumerate(rf_cycles.items(), 1):  # 각 싸이클에 대해
        ax = fig.add_subplot(2, len(rf_cycles), i + len(rf_cycles), projection='3d')  # 3D subplot 추가
        x_data = [cycle[0] for cycle in cycles]  # 싸이클 시작점
        y_data = [cycle[1] for cycle in cycles]  # 싸이클 끝점
        hist, x_edges, y_edges = np.histogram2d(x_data, y_data, bins=20)  # 2D 히스토그램 생성

        x_pos, y_pos = np.meshgrid(x_edges[:-1] + 0.25, y_edges[:-1] + 0.25)  # x, y 위치 설정
        x_pos = x_pos.flatten('F')  # 1차원 배열로 변환
        y_pos = y_pos.flatten('F')  # 1차원 배열로 변환
        z_pos = np.zeros_like(x_pos)  # z 위치는 0으로 설정
        dx = dy = 0.5 * np.ones_like(z_pos)  # 각 bin의 크기 설정
        dz = hist.flatten()  # 높이 설정

        # 3D 히스토그램 막대에 색상 그라디언트 적용
        dz_norm = dz - dz.min()  # 최소값을 빼서 0에서 시작하도록 함
        dz_norm /= dz_norm.ptp()  # 최대값으로 나눠서 0~1 사이로 정규화
        ax.bar3d(x_pos, y_pos, z_pos, dx, dy, dz, color=cmap(dz_norm), zsort='average')  # 3D 바 그래프 그리기

        ax.set_title(f"3D Histogram for {y_name}")  # 제목 설정
        ax.set_xlabel('Cycle Start')  # x축 레이블 설정
        ax.set_ylabel('Cycle End')  # y축 레이블 설정
        ax.set_zlabel('Count')  # z축 레이블 설정

    plt.tight_layout()  # 레이아웃 조정
    plt.show()  # 그래프 보여주기


# 다중 도수분포표를 생성하는 함수
def frequency_distribution(rf_cycles):
    fdf_list = [] # 저장할 다중 도수분포표 리스트를 초기화

    # 각 데이터 시리즈에 대해 도수분포표를 생성
    for y_name, cycles in rf_cycles.items():
        frequencies = [cycle[1] - cycle[0] for cycle in cycles]  # 각 사이클의 주파수를 계산
        bins = pd.cut(frequencies, bins=10)  # 계산된 주파수를 10개의 구간으로 나눔
        fdf = pd.DataFrame(pd.value_counts(bins, sort=False).reset_index().sort_values(by='index'))  # 각 구간에 대한 도수분포표를 생성하고, DataFrame으로 변환
        fdf.columns = ['Bin', 'Count']  # 컬럼 이름을 변경
        fdf['y_name'] = y_name  # y_name 열을 추가하여, 어떤 데이터 시리즈에 대한 도수분포표인지 표시
        fdf_list.append(fdf)  # 생성한 도수분포표를 리스트에 추가

    # 모든 도수분포표를 하나의 DataFrame으로 합침
    total_fdf = pd.concat(fdf_list, ignore_index=True)

    return total_fdf  # 최종적으로 생성한 다중 도수분포표를 반환


# 다중 도수분포표를 그리는 함수
def draw_table(fdf):
    window = tk.Tk()  # GUI 창을 생성

    # 스타일을 설정
    style = ttk.Style()
    style.configure('Treeview', rowheight=25)  # 행 높이를 조절
    style.configure('Treeview', font=('Arial', 12))  # 폰트 크기를 조절
    style.configure('Treeview.Heading', font=('Arial', 14, 'bold'))  # 헤더의 폰트를 설정

    # 각 도수분포표에 대해 표와 레이블을 생성
    for y_name in fdf['y_name'].unique():
        frame = tk.Frame(window)  # 새로운 프레임을 생성
        frame.pack(fill='x')  # 생성한 프레임을 창에 추가

        # 레이블을 생성
        label = tk.Label(frame, text=f"{y_name}에 대한 도수분포", font=('Arial', 16, 'bold'), bg='light gray')
        label.pack(fill='x')  # 생성한 레이블을 프레임에 추가

        # 표를 생성
        fdf_y = fdf[fdf['y_name'] == y_name]  # 현재 y_name에 해당하는 도수분포표를 가져옵니다.
        table = ttk.Treeview(frame, columns=list(fdf_y.columns), show='headings')  # 표를 생성하고, 컬럼을 설정
        for column in fdf_y.columns:  # 각 컬럼에 대해
            table.heading(column, text=column)  # 헤더를 설정
        for row in fdf_y.values:  # 각 행에 대해
            table.insert('', 'end', values=tuple(row))  # 행을 표에 추가
        table.pack()  # 생성한 표를 프레임에 추가

    window.mainloop()  # GUI를 실행

def main():
    # 유저 입력 받기
    #file_path, x_name, y_names = get_user_input()
    # 유저로부터 입력 받는 부분, 테스트를 위해 주석 처리

    # Input File (Temporary, for testing)
    file_path = "C:/Users/APTech-dev03/Documents/2. 프로젝트/KAI 제공자료/FA-50_VADR/test2.csv"
    # file_path = "C:/Users/Paul Kim/Documents/@_DOCS_@/VSCode Repo/데이터/하중2.csv"  # 테스트용 파일 경로 설정
    x_name = "TIME"  # x축의 이름을 "TIME"으로 설정
    y_names = ["NZ","LONGACCEL","NY"]  # y축의 이름을 "NZ", "LONGACCEL", "NY"로 설정

    # 아웃라이어 필터의 최소값과 최대값 설정
    filter_min = -100
    filter_max = 100

    # 이동평균 필터(MAF)의 윈도우 크기 설정
    window_size = 2
    
    # 칼만 필터의 설정값 지정
    P_value = 1000
    R_value = 5
    Q_value = 10
    
    # 데이터 전처리 수행
    df, x_units, y_units = process_data(file_path, x_name, y_names, filter_min, filter_max)

    # 원시 데이터를 csv 파일로 저장
    print("Saving raw data dictionary as csv:")
    output_to_csv(df, file_path, 'processed')
    
    # 이동평균 필터 적용 후 결과를 csv 파일로 저장
    print("Saving MAF data dictionary as csv:")
    mdf = apply_maf(df, y_names, window_size)
    mdf_percent = calculate_filtering_percentage(df, mdf, y_names)  # MAF 필터링 백분율 계산
    for y_name in y_names:
        print(f"For {y_name}, the data was filtered by approximately {mdf_percent[y_name]:.2f}% in MAF")
    output_to_csv(mdf, file_path, 'maf')

    # 칼만 필터 적용 후 결과를 csv 파일로 저장
    print("Saving Kalman data dictionary as csv:")
    kdf = apply_kalman(mdf, x_name, y_names, P_value, R_value, Q_value)
    kdf_percent = calculate_filtering_percentage(df, kdf, y_names)  # 칼만 필터 필터링 백분율 계산
    for y_name in y_names:
        print(f"For {y_name}, the data was filtered by approximately {kdf_percent[y_name]:.2f}% in Kalman filter")
    output_to_csv(kdf, file_path, 'kalman')

    # 레인플로우 카운팅 적용 후 결과를 csv 파일로 저장
    print("Saving Rainflow data dictionary as csv:")
    rdf = apply_rainflow(kdf, y_names)
    output_rf_cycles_to_csv(rdf, file_path, 'rainflow')

    # 도수분포표 생성 후 결과를 csv 파일로 저장
    print("Saving Frequency Distribution dictionary as csv:")
    fdf = frequency_distribution(rdf)
    output_to_csv(fdf, file_path, 'frequency')
    
    # 도수분포표 그래픽 인터페이스로 그림
    draw_table(fdf)

    # 데이터에 대한 그래프 그림
    plot_graphs(df, kdf, mdf, x_name, y_names, x_units, y_units, mdf_percent, kdf_percent)

    # 레인플로우 카운팅의 결과를 2D, 3D 히스토그램으로 출력
    draw_graphs(rdf)
    

# 메인 함수 호출
if __name__ == "__main__":
    main()
    