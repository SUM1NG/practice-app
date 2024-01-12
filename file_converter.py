import pandas as pd
import os

def convert_xlsx_to_csv(input_file_path):
    # xlsx 파일 읽기
    data_xlsx = pd.read_excel(input_file_path, engine='openpyxl')

    # csv 파일로 저장할 이름 설정
    output_file_name = os.path.splitext(os.path.basename(input_file_path))[0] + '.csv'
    output_file_path = os.path.join(os.path.dirname(input_file_path), output_file_name)

    # csv 파일로 저장
    data_xlsx.to_csv(output_file_path, index=False, encoding='utf-8')

    print(f"{os.path.basename(input_file_path)} 파일이 {output_file_name} 파일로 성공적으로 변환되었습니다.")

# 사용자에게 파일 경로 입력 요청
file_path = input("변환하고자 하는 xlsx 파일의 경로를 입력하세요 (파일 이름과 확장자 .xlsx 포함): ")

# 함수 호출
convert_xlsx_to_csv(file_path)