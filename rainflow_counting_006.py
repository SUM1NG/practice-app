import pandas as pd
import matplotlib.pyplot as plt
import datetime
import numpy as np
import os

# 파일 경로를 입력받아 csv 파일을 읽는 함수
def read_csv_file():
    file_path = input("csv 파일의 경로를 입력해주세요: ").replace("\"", "")
    df = pd.read_csv(file_path)
    return df, file_path

# X축의 이름을 입력받는 함수
def get_x_name():
    x_name = input("X축의 이름을 입력해주세요: ")
    return x_name

# y축의 여러 이름을 입력받는 함수
def get_y_names():
    y_names = input("Y축의 이름을 입력해주세요(쉼표로 구분): ").split(",")
    return y_names

# 새로운 csv 파일을 생성하는 함수
def create_new_csv(df, file_path, x_name, y_names):
    # 사용자가 지정하지 않은 열을 모두 삭제합니다
    df = df[[x_name] + y_names]

    # 새로운 csv 파일 이름을 생성합니다
    dir_name = os.path.dirname(file_path)
    new_file_path = os.path.join(dir_name, "[processed]" + os.path.basename(file_path))

    # 새로운 csv 파일을 만듭니다
    df.to_csv(new_file_path, index=False)

    return new_file_path

# "TIME"열의 데이터를 밀리세컨드로 변환하는 함수
def convert_time_to_ms(df, x_name):
    if x_name == "TIME":
        df[x_name] = df[x_name].apply(lambda x: datetime.datetime.strptime(x, '%H:%M:%S.%f').time())
        df[x_name] = df[x_name].apply(lambda x: (x.hour * 3600 + x.minute * 60 + x.second) * 1000 + x.microsecond / 1000)
    return df

# y축의 데이터를 float 형식으로 변환하는 함수
def convert_to_float(df, y_names):
    for y_name in y_names:
        if df[y_name].dtype != np.float:
            print(f"{y_name} 열의 데이터를 float 형식으로 변환합니다.")
            df[y_name] = df[y_name].astype(float)
    return df

# y축의 데이터가 반복되면 최소값으로 대체하는 함수
def replace_repeated_values_with_min(df, y_names):
    for y_name in y_names:
        min_val = df[y_name].min()
        df[y_name] = df[y_name].mask(df[y_name].duplicated(keep=False) & (df[y_name] != 0), min_val)
    return df

# y축의 데이터가 없는 경우 평균값으로 채우는 함수
def fill_blank_with_average(df, y_names):
    for y_name in y_names:
        df[y_name].fillna((df[y_name].shift() + df[y_name].shift(-1)) / 2, inplace=True)
    return df

# 새로운 csv 파일로부터 딕셔너리를 생성하는 함수
def create_dict_from_csv(new_file_path, x_name, y_names):
    df = pd.read_csv(new_file_path)

    data_dict = {}
    data_dict[x_name] = df[x_name].tolist()
    for y_name in y_names:
        data_dict[y_name] = df[y_name].tolist()

    return data_dict

# 그래프를 그리는 함수
def draw_graph(data_dict, x_name, y_names):
    color_list = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    fig, axs = plt.subplots(len(y_names), 1, figsize=(12, 8))
    x_data = []
    y_data = {y_name: [] for y_name in y_names}
    
    for x, *ys in zip(data_dict[x_name], *[data_dict[y_name] for y_name in y_names]):
        x_data.append(datetime.datetime.fromtimestamp(x/1000.0).strftime('%H:%M'))
        for y_name, y in zip(y_names, ys):
            y_data[y_name].append(y)

    for i, y_name in enumerate(y_names):
        axs[i].plot(x_data, y_data[y_name], color_list[i % len(color_list)], label=y_name)
        axs[i].legend(loc='upper center')
        axs[i].grid(True)
        axs[i].yaxis.set_major_locator(plt.MaxNLocator(15))
        axs[i].xaxis.set_major_locator(plt.MaxNLocator(20))

    fig.tight_layout()
    plt.show()

# 메인 함수
def main():
    df, file_path = read_csv_file()
    x_name = get_x_name()
    y_names = get_y_names()
    new_file_path = create_new_csv(df, file_path, x_name, y_names)
    df = convert_time_to_ms(df, x_name)
    df = convert_to_float(df, y_names)
    df = replace_repeated_values_with_min(df, y_names)
    df = fill_blank_with_average(df, y_names)
    data_dict = create_dict_from_csv(new_file_path, x_name, y_names)
    draw_graph(data_dict, x_name, y_names)

if __name__ == "__main__":
    main()
