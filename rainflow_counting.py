# 제목: Rainflow Counting
# 작성자: 김주형
# 제정: 2024/01/10
# 개정: 2024/01/12

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import find_peaks


# xlsx to csv
def convert_to_csv(input_file_path):
    try:
        # xlsx 파일 읽기
        data_xlsx = pd.read_excel(input_file_path, engine='openpyxl', header=0)

        # csv 파일로 저장할 이름 설정
        output_file_name = os.path.splitext(os.path.basename(input_file_path))[0] + '.csv'
        output_file_path = os.path.join(os.path.dirname(input_file_path), output_file_name)

        # csv 파일로 저장
        data_xlsx.to_csv(output_file_path, index=False, encoding='utf-8')

        print(f"{os.path.basename(input_file_path)} 파일이 {output_file_name} 파일로 성공적으로 변환되었습니다.")
        return output_file_path

    except Exception as e:
        print(f"파일 변환 중 오류가 발생했습니다: {e}")
        return None
    
# read csv
def read_csv(file_path):
    try:
        # 사용자에게 x축과 y축 입력 요청
        x_name = input("x축으로 사용할 데이터 열 이름을 입력하세요: ")
        y_name = input("y축으로 사용할 데이터 열 이름을 입력하세요 (여러 개일 경우 쉼표로 구분): ").split(',')

        # 단위 정보 가져오기
        units = pd.read_csv(file_path, header=None, skiprows=1, nrows=1)
        units.columns = pd.read_csv(file_path, nrows=1).columns
        
        # csv 파일 읽기 (단위 정보가 포함된 두 번째 행 제거)
        data_read = pd.read_csv(file_path, header=0, skiprows=1)



        x_axis = data_read[x_name]
        y_axis = data_read[y_name]

        return x_axis, y_axis

    except Exception as e:
        print(f"파일 읽기 중 오류가 발생했습니다: {e}")
        return None

# Rainflow algorithm
def rainflow(data):
    stack = []
    cycles = []
    for x in data:
        while len(stack) >= 2:
            [x1, x2, x3] = [stack[-2], stack[-1], x]
            if x1 <= x2 >= x3 or x1 >= x2 <= x3:
                if len(stack) >= 3 and (stack[-3] < x2 > x or stack[-3] > x2 < x):
                    break
                cycles.append((x1,x2))
                stack.pop(-1)
            else:
                break
        stack.append(x)
    for i in range(len(stack) - 1):
        cycles.append((stack[i], stack[i+1]))
    return cycles


# Distribution chart
def draw_distribution_chart(aaaa):
    try:
        fig, ax = plt.subplots()

        for y_axis in y_axes:
            # 결과를 분포 차트에 표시
            ax.plot(peaks, label=f"{y_axis} ({units[y_axis]})")

        ax.set_title('Distribution Chart')
        ax.legend()
        plt.show()

    except Exception as e:
        print(f"분포 차트 그리기 중 오류가 발생했습니다: {e}")

# 3D Histogram
def draw_3d_histogram(aaaa):
    try:
        fig_3d = plt.figure()
        ax_3d = fig_3d.add_subplot(111, projection='3d')

        for y_axis in y_axes:

            # 결과를 3D 히스"C:\Users\APTech-dev03\Documents\2. 프로젝트\KAI 제공자료\FA-50_VADR\FA-50_VADR_Nxyz_01.csv"토그램으로 표시
            hist, xedges, yedges = np.histogram2d(data[x_axis], peaks, bins=10, range=[[0, 1], [0, 1]])
            ax_3d.bar3d(xedges[:-1], yedges[:-1], np.zeros(len(xedges)-1), 1, 1, hist.flatten(), shade=True)
            ax_3d.set_ylabel(f"{y_axis} ({units[y_axis]})")

        ax_3d.set_title('3D Histogram')
        plt.show()

    except Exception as e:
        print(f"3D 히스토그램 그리기 중 오류가 발생했습니다: {e}")

# Main function
def main():
    # 사용자에게 파일 경로 입력 요청
    file_path = input("처리하고자 하는 파일의 경로를 입력하세요 (파일 이름과 확장자 포함): ")

    # 파일이 csv가 아니면 변환
    if not file_path.endswith('.csv'):
        file_path = convert_to_csv(file_path)

    if file_path is not None:
        # csv 파일 읽기 
        data = read_csv(file_path)
        
        rainflow()
        # Rainflow Counting 수행
        
        # 분포 차트 그리기
        draw_distribution_chart(aaaa)

        # 3D 히스토그램 그리기
        draw_3d_histogram(aaaa)

# Main function 실행
main()
