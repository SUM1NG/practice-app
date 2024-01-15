import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from math import ceil

# 파일 경로를 입력받아 CSV 파일 읽기
def read_csv_file():
    file_path = input("CSV 파일 경로를 입력하세요: ").replace("\"", "")
    try:
        df = pd.read_csv(file_path)
        print("CSV 파일을 성공적으로 읽었습니다.")
        return df, file_path
    except Exception as e:
        print(f"CSV 파일을 읽는 중 오류가 발생했습니다: {e}")
        return None, None

# x축의 이름을 입력받기
def get_x_name():
    x_name = input("x축의 이름을 입력하세요: ")
    print(f"x축의 이름을 {x_name}으로 설정합니다.")
    return x_name

# y축의 이름들을 쉼표로 구분하여 입력받기
def get_y_names():
    y_names = input("y축의 이름들을 쉼표로 구분하여 입력하세요: ").split(",")
    print(f"y축의 이름들을 {y_names}으로 설정합니다.")
    return y_names

# 새로운 CSV 파일 생성
def create_new_csv(df, file_path, x_name, y_names):
    # 새로운 CSV 파일의 이름 설정
    new_file_name = "[processed]" + os.path.basename(file_path)
    new_file_path = os.path.join(os.path.dirname(file_path), new_file_name)

    # 사용자가 지정한 열만 남기기
    df = df[[x_name] + y_names]
    print(f"데이터프레임의 열을 {x_name}, {y_names}만 남기도록 수정했습니다.")

    # 공백 제거
    df = df.replace(r'^\s*$', np.nan, regex=True)
    print("데이터프레임의 모든 공백을 제거했습니다.")

    # 열 순서 재배열
    df = df.reindex(columns=[x_name] + y_names)
    print(f"데이터프레임의 열 순서를 {x_name}, {y_names} 순으로 재배열했습니다.")

    # 단위 저장 후 첫 두 행 삭제
    x_units = df[x_name][1]
    y_units = {y_name: df[y_name][1] for y_name in y_names}
    df = df.iloc[2:]
    print("데이터프레임의 첫 두 행을 삭제했습니다.")

    # CSV 파일 저장
    df.to_csv(new_file_path, index=False)
    print(f"새로운 CSV 파일을 {new_file_path}에 저장했습니다.")

    return df, new_file_path, x_units, y_units

# 시간을 밀리세컨드로 변환
def convert_time_to_millis(df, x_name):
    df[x_name] = df[x_name].apply(lambda x: sum(float(y) * 10 ** i for i, y in enumerate(reversed(x.split(":")))) * 1000)
    print(f"{x_name} 열의 시간을 밀리세컨드로 변환했습니다.")
    return df

# 연속적인 중복 값 처리
def handle_consecutive_duplicates(df, y_names):
    for y_name in y_names:
        df[y_name] = df[y_name].where(~(df[y_name] == df[y_name].shift(1) == df[y_name].shift(2)) | (df[y_name] == 0), df[y_name].min())
    print(f"{y_names} 열의 연속적인 중복 값들을 처리했습니다.")
    return df

# 빈 칸 채우기
def fill_blanks(df, y_names):
    for y_name in y_names:
        df[y_name].fillna(df[y_name].rolling(2, min_periods=1).mean(), inplace=True)
    print(f"{y_names} 열의 빈 칸들을 채웠습니다.")
    return df

# 데이터 딕셔너리 생성
def create_data_dict(df, x_name, y_names, x_units, y_units):
    data_dict = {x_name: {'units': x_units, 'data': df[x_name].tolist()}}
    for y_name in y_names:
        data_dict[y_name] = {'units': y_units[y_name], 'data': df[y_name].tolist()}
    print("데이터 딕셔너리를 생성했습니다.")
    return data_dict

# 그래프 그리기
def draw_graph(data_dict, x_name, y_names):
    fig, axs = plt.subplots(1, len(y_names), figsize=(15, 5))
    for i, y_name in enumerate(y_names):
        axs[i].plot(data_dict[x_name]['data'], data_dict[y_name]['data'])
        axs[i].set_title(y_name)
        axs[i].set_xlabel(x_name)
        axs[i].set_ylabel(y_name)
        axs[i].set_xticks(np.linspace(min(data_dict[x_name]['data']), max(data_dict[x_name]['data']), 20))
        axs[i].set_yticks(np.linspace(min(data_dict[y_name]['data']), max(data_dict[y_name]['data']), 15))
    plt.tight_layout()
    plt.show()
    print("그래프를 그렸습니다.")

def main():
    df, file_path = read_csv_file()
    if df is None:
        return
    x_name = get_x_name()
    y_names = get_y_names()
    df, new_file_path, x_units, y_units = create_new_csv(df, file_path, x_name, y_names)
    df = convert_time_to_millis(df, x_name)
    df = handle_consecutive_duplicates(df, y_names)
    df = fill_blanks(df, y_names)
    data_dict = create_data_dict(df, x_name, y_names, x_units, y_units)
    draw_graph(data_dict, x_name, y_names)

if __name__ == "__main__":
    main()
