import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
#from datetime import datetime
from matplotlib.ticker import MaxNLocator

# 파일 경로를 입력받아 csv 파일 읽는 함수
def read_csv_file():
    # 파일 경로 입력 받기
    file_path = input('CSV 파일 경로를 입력해주세요: ').strip('\"')
    # csv 파일 읽기
    df = pd.read_csv(file_path)
    return df, file_path

# x축, y축 이름을 입력받는 함수
def get_axis_names():
    # x축 이름 입력 받기
    x_name = input('x축의 이름을 입력해주세요: ')
    # y축 이름 입력 받기
    y_names = input('y축의 이름을 입력해주세요(콤마로 구분): ').split(',')
    return x_name, y_names

# 시간 문자열(HH:MM:SS.FFF)을 밀리초로 변환하는 함수
def time_str_to_milliseconds(time_str):
    h, m, rest = time_str.split(':')
    s, ms = map(float, rest.split('.'))
    milliseconds = ((int(h) * 60 + int(m)) * 60 + s) * 1000 + ms
    return milliseconds

# 데이터 처리하는 함수
def process_data(df, x_name, y_names):
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
        df[y_name].interpolate(method='linear', limit_direction ='forward', inplace=True)

    return df, x_units, y_units


# 데이터를 그래프로 그리는 함수
def draw_graphs(df, x_name, y_names, x_units, y_units):
    fig, axs = plt.subplots(len(y_names), sharex=True)
    fig.suptitle('Graphs')
    for i, y_name in enumerate(y_names):
        axs[i].plot(df[x_name], df[y_name])
        axs[i].set_title(y_name)
        axs[i].yaxis.set_major_locator(MaxNLocator(15))
        axs[i].xaxis.set_major_locator(MaxNLocator(20))
        #axs[i].xaxis.set_major_formatter(lambda x, _: str(datetime.timedelta(milliseconds=x)))
    plt.show()

def main():
    # CSV 파일 읽기
    df, file_path = read_csv_file()

    # x축, y축 이름 얻기
    x_name, y_names = get_axis_names()

    # 데이터 처리
    df, x_units, y_units = process_data(df, x_name, y_names)

    # 새로운 CSV 파일 생성
    df.to_csv(os.path.join(os.path.dirname(file_path), f'[processed]{os.path.basename(file_path)}'), index=False)

    # 그래프 그리기
    draw_graphs(df, x_name, y_names, x_units, y_units)

if __name__ == '__main__':
    main()
