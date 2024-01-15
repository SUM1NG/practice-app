# 필요한 라이브러리 임포트
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# 시간 문자열(HH:MM:SS.FFF)을 밀리초로 변환하는 함수
def time_str_to_milliseconds(time_str):
    """
    시간 문자열 milisecond로 변환
    입력: 시간 문자열(HH:MM:SS.FFF 형식, string)
    출력: milisecond(float)
    """
    h, m, rest = time_str.split(':')
    s, ms = map(float, rest.split('.'))
    milliseconds = ((int(h) * 60 + int(m)) * 60 + s) * 1000 + ms
    return milliseconds

# 데이터 처리하는 함수
def process_data(file_path, x_name, y_names):
    """
    CSV 파일 읽고 데이터 처리
    입력: 파일 경로, x 축 이름, y 축 이름(list)
    출력: 처리된 데이터프레임, x 축 단위, y 축 단위(list)
    """
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
    """
    # 동일한 값이 2번 이상 연속으로 나오는 경우 최소값으로 대체
    for y_name in y_names:
       df[y_name] = df[y_name].mask(df[y_name].eq(df[y_name].shift()) & df[y_name].eq(df[y_name].shift(-1)) & df[y_name].ne(0), df[y_name].min())
    """
    # 중복 값을 찾아 NaN으로 대체
    for y_name in y_names:
        df.loc[df[y_name].eq(df[y_name].shift()) & df[y_name].eq(df[y_name].shift(-1)) & df[y_name].ne(0), y_name] = np.nan

    # 중복 값이 NaN으로 대체된 후, 해당 열에서 두 번째로 작은 값을 찾아 대체
    for y_name in y_names:
        second_min = df[y_name].nsmallest(2).values[-1]
        df[y_name].fillna(second_min, inplace=True)

    # 빈 값은 앞뒤 값의 평균으로 대체
    for y_name in y_names:
        df[y_name].interpolate(method='linear', limit_direction ='forward', inplace=True)

    return df, x_units, y_units

# 데이터 그래프 그리는 함수
def plot_data(df, x_name, y_names, x_units, y_units):
    """
    데이터프레임을 이용해 그래프를 그립니다.
    입력: 데이터프레임, x 축 이름, y 축 이름들(list), x 축 단위, y 축 단위들(list)
    출력: 없음
    """
    # 그래프를 세로로 쌓음
    fig, axs = plt.subplots(len(y_names), 1, figsize=(35, 15))

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

def main():
    """
    메인 함수입니다. 사용자로부터 필요한 정보를 입력받고, 데이터 처리 및 그래프 그리기를 수행합니다.
    입력: 없음
    출력: 없음
    """
    # 사용자로부터 필요한 정보 입력받기
    file_path = input("Enter the file path: ").replace('"', '')
    x_name = input("Enter the x-axis name: ").strip() # x_name에서 공백 제거
    y_names = [name.strip() for name in input("Enter the y-axis names (separated by comma): ").split(',')] # y_names에서 공백 제거

    # 데이터 처리
    df, x_units, y_units = process_data(file_path, x_name, y_names)

    # 새로운 CSV 파일 생성
    new_file_path = os.path.join(os.path.dirname(file_path), f"[processed]{os.path.basename(file_path)}")
    df.to_csv(new_file_path, index=False)

    # 그래프 그리기
    plot_data(df, x_name, y_names, x_units, y_units)

if __name__ == "__main__":
    main()
