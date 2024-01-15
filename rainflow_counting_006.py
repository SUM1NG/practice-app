# 필요한 라이브러리를 불러옵니다.
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import datetime

def get_inputs():
    """
    사용자로부터 파일 경로와 x축, y축의 이름을 입력 받습니다.
    입력: 없음
    출력: 파일 경로, x축 이름, y축 이름 리스트
    """
    file_path = input("CSV 파일 경로를 입력해주세요: ").replace('"', '')
    x_name = input("x축의 이름을 입력해주세요: ")
    y_names = input("y축의 이름을 입력해주세요(쉼표로 구분): ").split(',')
    return file_path, x_name, y_names

def process_df(file_path, x_name, y_names):
    """
    CSV 파일을 읽고, 사용자가 지정한 열만 남기고 모두 삭제합니다. 
    또한, 열의 순서를 조정하고, 빈 공간을 제거합니다.
    입력: 파일 경로, x축 이름, y축 이름 리스트
    출력: 처리된 DataFrame, x축 단위, y축 단위 리스트
    """
    df = pd.read_csv(file_path)
    x_units = df.loc[1, x_name]
    y_units = [df.loc[1, y_name] for y_name in y_names]

    # 사용자가 지정하지 않은 열을 제거합니다.
    df = df[[x_name] + y_names]

    # 빈 공간을 제거합니다.
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

    # 열의 순서를 조정합니다.
    df = df[[x_name] + y_names]

    # x_name, y_name, x_units, y_units가 포함된 행을 제거합니다.
    df = df[df[x_name] != x_name]
    df = df[df[x_name] != x_units]

    return df, x_units, y_units

def to_milliseconds(time_str):
    """
    시간 문자열(HH:MM:SS.FFF)을 밀리초로 변환합니다.
    입력: 시간 문자열
    출력: 밀리초(float)
    """
    h, m, s = map(float, time_str.split(':'))
    return h*3600000 + m*60000 + s*1000

def process_data(df, x_name, y_names):
    """
    데이터를 처리합니다. 
    x축 데이터를 밀리초로 변환하고, y축 데이터를 float으로 변환합니다.
    입력: DataFrame, x축 이름, y축 이름 리스트
    출력: 처리된 DataFrame
    """
    # x축 데이터를 밀리초로 변환합니다.
    df[x_name] = df[x_name].apply(to_milliseconds)

    # y축 데이터를 float으로 변환합니다.
    for y_name in y_names:
        df[y_name] = df[y_name].astype(float)

    return df

def remove_duplicates(df, y_names):
    """
    y축에서 중복된 값을 제거합니다.
    입력: DataFrame, y축 이름 리스트
    출력: 중복된 값이 제거된 DataFrame
    """
    for y_name in y_names:
        df.loc[df[y_name].duplicated(keep=False) & df[y_name]!=0, y_name] = df[y_name].min()
    return df

def fill_blanks(df, y_names):
    """
    y축에서 빈 값을 채웁니다.
    입력: DataFrame, y축 이름 리스트
    출력: 빈 값이 채워진 DataFrame
    """
    for y_name in y_names:
        df[y_name] = df[y_name].fillna((df[y_name].shift() + df[y_name].shift(-1))/2)
        df[y_name].fillna(method='ffill', inplace=True)
        df[y_name].fillna(method='bfill', inplace=True)
    return df

def create_dict(df, x_name, y_names, x_units, y_units):
    """
    DataFrame을 딕셔너리로 변환합니다.
    입력: DataFrame, x축 이름, y축 이름 리스트, x축 단위, y축 단위 리스트
    출력: 데이터 딕셔너리
    """
    data_dict = {x_name: {'units': x_units, 'data': df[x_name].tolist()}}
    for y_name, y_unit in zip(y_names, y_units):
        data_dict[y_name] = {'units': y_unit, 'data': df[y_name].tolist()}
    return data_dict

def draw_graphs(data_dict, x_name, y_names):
    """
    데이터를 그래프로 그립니다.
    입력: 데이터 딕셔너리, x축 이름, y축 이름 리스트
    출력: 없음
    """
    fig, axs = plt.subplots(len(y_names), figsize=(10, 5*len(y_names)), sharex=True)
    fig.subplots_adjust(hspace=0.5)

    x_data = data_dict[x_name]['data']

    for ax, y_name in zip(axs, y_names):
        y_data = data_dict[y_name]['data']
        ax.plot(x_data, y_data)
        ax.set_title(y_name, loc='center')
        ax.set_ylabel(y_name)
        ax.set_xlabel(x_name)
        ax.set_yticks(np.linspace(min(y_data), max(y_data), 15))
        ax.set_xticks(np.linspace(min(x_data), max(x_data), 20))
        ax.grid()

    plt.tight_layout()
    plt.show()

def main():
    file_path, x_name, y_names = get_inputs()
    df, x_units, y_units = process_df(file_path, x_name, y_names)
    df = process_data(df, x_name, y_names)
    df = remove_duplicates(df, y_names)
    df = fill_blanks(df, y_names)
    data_dict = create_dict(df, x_name, y_names, x_units, y_units)
    draw_graphs(data_dict, x_name, y_names)

if __name__ == "__main__":
    main()
