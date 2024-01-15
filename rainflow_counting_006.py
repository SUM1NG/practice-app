import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import timedelta


def read_csv_file():
    """
    이 함수는 사용자로부터 CSV 파일 경로를 입력받아 해당 파일을 읽어옵니다.
    입력: 없음
    출력: CSV 파일의 내용(DataFrame)
    """
    file_path = input("CSV 파일 경로를 입력해주세요: ").replace('"', '')
    df = pd.read_csv(file_path)
    return df


def get_axis_names():
    """
    이 함수는 사용자로부터 x축 이름과 y축 이름들을 입력받습니다.
    입력: 없음
    출력: x축 이름(string), y축 이름들(list of string)
    """
    x_name = input("x축의 이름을 입력해주세요: ")
    y_names = input("y축의 이름을 입력해주세요(쉼표로 구분): ").split(',')
    return x_name, y_names


def process_data(df, x_name, y_names):
    """
    이 함수는 입력된 DataFrame의 데이터를 처리하여 새로운 DataFrame을 생성합니다.
    입력: 원본 DataFrame, x축 이름(string), y축 이름들(list of string)
    출력: 처리된 DataFrame
    """
    # 사용자가 지정한 열만 남기고 모든 열 삭제
    df = df[[x_name] + y_names]

    # 빈칸 삭제
    df = df.dropna()

    # x_name의 데이터를 밀리초로 변환
    df[x_name] = df[x_name].apply(to_milliseconds)

    # y_name의 데이터를 float로 변환
    for y_name in y_names:
        df[y_name] = df[y_name].astype(float)

    return df


def to_milliseconds(time_str):
    """
    이 함수는 주어진 시간 문자열(HH:MM:SS.FFF)을 밀리초(float)로 변환합니다.
    입력: 시간 문자열(string)
    출력: 밀리초(float)
    """
    h, m, s = time_str.split(':')
    s, ms = map(float, s.split('.'))
    return (int(h) * 3600 + int(m) * 60 + s + ms/1000) * 1000


def main():
    # CSV 파일 읽기
    df = read_csv_file()

    # x축, y축 이름 얻기
    x_name, y_names = get_axis_names()

    # 데이터 처리
    df = process_data(df, x_name, y_names)

    # 새로운 CSV 파일 생성
    df.to_csv(f'[processed]{os.path.basename(file_path)}', index=False)


if __name__ == "__main__":
    main()
