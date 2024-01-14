import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import csv

def get_input(prompt):
    return input(prompt)

def read_file(file_path):
    print("파일을 읽는 중입니다.")
    try:
        data = pd.read_csv(file_path, encoding='utf-8')
        print("파일이 성공적으로 불러와졌습니다.")
    except UnicodeDecodeError:
        print("UTF-8로 해석할 수 없는 문자가 있습니다. 해당 문자를 제거합니다.")
        with open(file_path, 'r', errors='ignore') as f:
            data = pd.read_csv(f)
    except Exception as e:
        print(f"파일을 불러오는 중 에러가 발생했습니다: {e}")
        return
    return data

def process_file(data, x_name, y_names, file_path):
    print("파일 처리를 시작합니다.")
    data.columns = data.iloc[0]
    data = data[1:]
    data.reset_index(drop=True, inplace=True)
    data.columns.name = None
    data = data.loc[:, data.columns.intersection([x_name] + y_names)]
    new_file_path = os.path.join(os.path.dirname(file_path), '[processed]' + os.path.basename(file_path))
    data.to_csv(new_file_path, index=False)
    print(f"처리된 파일이 '{new_file_path}'에 저장되었습니다.")
    return data, new_file_path

def calculate_sampling_rate(data, y_names, new_file_path):
    print("샘플링 비율 계산을 시작합니다.")
    sampling_rate = {0: 16, 1: 8, 3: 4, 7: 2, 15: 1}
    new_data = pd.read_csv(new_file_path)
    for y in y_names:
        blanks = data[y].isna().sum()
        if blanks in sampling_rate:
            print(f"{y}의 샘플링 비율은 {sampling_rate[blanks]}입니다.")
            new_data.loc[1, y] = sampling_rate[blanks]
        else:
            print(f"{y}의 샘플링 비율 계산 중 에러가 발생했습니다.")
    new_data.to_csv(new_file_path, index=False)
    return new_data

def process_values(data, y_names, new_file_path):
    print("값 처리를 시작합니다.")
    for y in y_names:
        min_val = data[y].min()
        data[y] = data[y].apply(lambda x: min_val if x == 0 else x)
        data[y].fillna(data[y].mean(), inplace=True)
    data.to_csv(new_file_path, index=False)
    return data

def create_dict(data, x_name, y_names):
    print("사전 생성을 시작합니다.")
    data_dict = {}
    for y in y_names:
        data_dict[y] = {"unit": data[y].name, 
                        "sampling_rate": data.loc[1, y], 
                        "data": data[y].tolist()}
    data_dict[x_name] = {"unit": data[x_name].name, 
                         "sampling_rate": data.loc[1, x_name], 
                         "data": data[x_name].tolist()}
    return data_dict

def draw_graph(data_dict, x_name, y_names):
    print("그래프 그리기를 시작합니다.")
    plt.figure(figsize=(8, 4))
    for y in y_names:
        plt.plot(data_dict[x_name]["data"], data_dict[y]["data"], label=y)
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.yticks(np.linspace(min(min(data_dict[x_name]["data"]), 
                             min(min([data_dict[y]["data"] for y in y_names]))), 
                           max(max(data_dict[x_name]["data"]), 
                               max(max([data_dict[y]["data"] for y in y_names]))), 15))
    plt.xticks(np.linspace(min(data_dict[x_name]["data"]), 
                           max(data_dict[x_name]["data"]), 20))
    plt.show()

def main():
    file_path = get_input("파일 경로를 입력하세요: ").strip('\"')
    x_name = get_input("x축의 이름을 입력하세요: ")
    y_names = get_input("y축의 이름을 입력하세요 (','로 구분): ").split(',')
    data = read_file(file_path)
    data, new_file_path = process_file(data, x_name, y_names, file_path)
    new_data = calculate_sampling_rate(data, y_names, new_file_path)
    new_data = process_values(new_data, y_names, new_file_path)
    data_dict = create_dict(new_data, x_name, y_names)
    draw_graph(data_dict, x_name, y_names)

if __name__ == "__main__":
    main()
