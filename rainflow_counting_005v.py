import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def process_csv(file_path, x_name, y_names):
    """
    간단한 설명: CSV 파일을 처리하여 새로운 CSV 파일을 생성합니다.
    입력: file_path(str) - CSV 파일의 전체 경로.
           x_name(str) - X축으로 사용할 열의 이름.
           y_names(list) - Y축으로 사용할 열의 이름.
    출력: new_file_path(str) - 새로 생성된 CSV 파일의 전체 경로.
    """
    print("CSV 파일을 처리하는 중입니다...")

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"파일을 읽는 중 에러가 발생했습니다: {e}")
        return

    # 사용자가 지정한 열만 선택합니다
    df = df[[x_name] + y_names] 

    # 새 CSV 파일을 만듭니다
    new_file_path = file_path.replace(".csv", "[processed].csv")
    df.to_csv(new_file_path, index=False)

    print(f"{new_file_path}에 새로운 CSV 파일이 생성되었습니다.")
    return new_file_path

def convert_time_to_seconds(time_str):
    """
    간단한 설명: 'HH:MM:SS.FFF' 형식의 시간 문자열을 밀리초로 변환합니다.
    입력: time_str(str) - 'HH:MM:SS.FFF' 형식의 시간 문자열.
    출력: total_milliseconds(float) - 밀리초 단위로 변환된 시간.
    """
    try:
        hours, minutes, seconds = map(float, time_str.split(':'))
        total_milliseconds = (hours * 3600 + minutes * 60 + seconds) * 1000
    except Exception as e:
        print(f"시간을 밀리초로 변환하는 중 에러가 발생했습니다: {e}")
        return

    return total_milliseconds

def create_data_dict(file_path, x_name, y_names):
    """
    간단한 설명: CSV 파일에서 딕셔너리를 생성합니다.
    입력: file_path(str) - CSV 파일의 전체 경로.
           x_name(str) - X축으로 사용할 열의 이름.
           y_names(list) - Y축으로 사용할 열의 이름.
    출력: data_dict(dict) - 지정된 데이터를 포함하는 딕셔너리.
    """
    print("딕셔너리를 생성하는 중입니다...")

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"파일을 읽는 중 에러가 발생했습니다: {e}")
        return

    data_dict = {}

    # X축 데이터를 딕셔너리에 추가합니다
    if x_name == 'TIME':
        print("시간을 밀리초로 변환하는 중입니다...")
        data_dict[x_name] = df[x_name].apply(convert_time_to_seconds).tolist()
    else:
        data_dict[x_name] = df[x_name].tolist()

    # Y축 데이터를 딕셔너리에 추가합니다
    for y_name in y_names:
        data_dict[y_name] = df[y_name].tolist()

    print("딕셔너리 생성이 완료되었습니다.")
    return data_dict

def draw_graph(data_dict, x_name, y_names):
    """
    간단한 설명: 딕셔너리 데이터를 사용해 그래프를 그립니다.
    입력: data_dict(dict) - 플롯할 데이터를 포함하는 딕셔너리.
           x_name(str) - X축으로 사용할 열의 이름.
           y_names(list) - Y축으로 사용할 열의 이름.
    출력: 없음
    """
    print("그래프를 그리는 중입니다...")

    try:
        plt.figure(figsize=(16, 8))

        # 각 Y축 데이터에 대해 그래프를 그립니다
        for y_name in y_names:
            plt.plot(data_dict[x_name], data_dict[y_name], label=y_name)

        plt.legend(loc='upper right')
        plt.grid(True)
        plt.show()

        print("그래프 그리기가 완료되었습니다.")
    except Exception as e:
        print(f"그래프를 그리는 중 에러가 발생했습니다: {e}")

def main():
    """
    간단한 설명: 프로그램을 실행하는 메인 함수.
    입력: 없음
    출력: 없음
    """
    # 사용자 입력을 받습니다
    file_path = input("파일 경로를 입력해주세요: ").replace("\"", "")
    x_name = input("x축의 이름을 입력해주세요: ")
    y_names = input("y축의 이름을 콤마로 구분하여 입력해주세요: ").split(',')

    # CSV 파일을 처리합니다
    new_file_path = process_csv(file_path, x_name, y_names)

    # 딕셔너리를 만듭니다
    data_dict = create_data_dict(new_file_path, x_name, y_names)

    # 그래프를 그립니다
    draw_graph(data_dict, x_name, y_names)

if __name__ == "__main__":
    main()
