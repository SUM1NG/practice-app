# 필요한 라이브러리 임포트
import os

# 필요한 패키지들이 설치되어 있는지 확인하고, 없다면 설치
try:
    import pandas as pd
except ImportError:
    os.system('pip install pandas')

"""
# 사용자로부터 필요한 정보 입력받기
def get_user_input():
    file_path = input("Enter the file path: ").replace('"', '')  # 파일 경로 입력 받음
    x_name = input("Enter the x-axis name: ").strip()  # x축 이름 입력 받음
    y_names = [name.strip() for name in input("Enter the y-axis names (separated by comma): ").split(',')]  # y축 이름들 입력 받음
    return file_path, x_name, y_names  # 입력 받은 정보 반환
"""

# 데이터 처리하는 함수
def process_data(file_path, x_name, y_names):
    # 데이터 읽기
    df = pd.read_csv(file_path)  # csv 파일 읽어와서 DataFrame 생성

    # 불필요한 열 삭제
    df = df[[x_name] + y_names]  # 필요한 열만 선택해서 DataFrame 재생성

    # 공백 제거
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)  # 각 열의 값이 문자열인 경우, 앞뒤 공백 제거

    # 열 순서 재배치
    df = df[[x_name] + y_names]  # 열 순서를 [x_name, y_names] 순서로 재배치

    # 단위 저장 후 해당 행 삭제
    df = df.drop([0])  # 단위를 저장한 첫 번째 행 삭제
    df = df.reset_index(drop=True)  # 인덱스 재설정

    # 공백 제거
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)  # 각 열의 값이 문자열인 경우, 앞뒤 공백 제거

    # HH:MM:SS.FFF를 밀리초로 변환
    df[x_name] = df[x_name].apply(time_str_to_milliseconds)  # 시간 문자열을 밀리초로 변환하는 함수 적용

    # 데이터 타입 확인 및 변환
    df = df.astype(float, errors='raise')  # 각 열의 값들을 실수로 변환

    # 동일한 `x_name` 값을 가진 행 제거
    df = df.drop_duplicates(subset=[x_name], keep='first')  # x_name 값이 동일한 행 중 첫 번째 행만 남기고 삭제

    # 빈 값은 앞뒤 값의 평균으로 대체
    for y_name in y_names:  # 각 y_name에 대해
        df[y_name].interpolate(method='linear', limit_direction ='both', inplace=True)  # 빈 값이 있는 경우, 앞뒤 값의 선형적인 평균으로 대체

    return df
    

# 결과를 CSV 파일로 출력하는 함수
def output_to_csv(df, file_path, prefix):
    new_file_path = os.path.join(os.path.dirname(file_path), f"[{prefix}]{os.path.basename(file_path)}")  # 새 파일 경로 생성
    df.to_csv(new_file_path, index=False)  # DataFrame을 csv 파일로 저장
    

# 시간 문자열(HH:MM:SS.FFF)을 밀리초로 변환하는 함수
def time_str_to_milliseconds(time_str):
    # 예외 처리 추가
    try:
        h, m, rest = time_str.split(':')  # 시간 문자열을 시, 분, 나머지로 분리
        s, ms = map(float, rest.split('.'))  # 나머지를 초, 밀리초로 분리
        milliseconds = ((int(h) * 60 + int(m)) * 60 + s) * 1000 + ms  # 밀리초로 변환
    except ValueError:  # 형식이 맞지 않는 경우 예외 처리
        print("오류: 시간 문자열이 'HH:MM:SS.FFF' 형식에 맞지 않습니다.")
        return None
    return milliseconds  # 밀리초 반환

def main():
    # 유저 입력 받기
    #file_path, x_name, y_names = get_user_input()
    # 유저로부터 입력 받는 부분, 테스트를 위해 주석 처리

    # Input File (Temporary, for testing)
    file_path = "C:/Users/APTech-dev03/Documents/2. 프로젝트/Acts Tech 제공할 자료/50-004_230414/50-004_2304141.csv"

    # file_path = "C:/Users/Paul Kim/Documents/@_DOCS_@/VSCode Repo/데이터/하중2.csv"  # 테스트용 파일 경로 설정
    x_name = "TIME"  # x축의 이름을 "TIME"으로 설정
    y_names = ["PRESALT", "CAS", "N2", "N1", "PLA", "TET", "T1", "VENACTSTKE", "TFAT", "STATPRES", "FVG", "CVG"]  # y축 이름 설정

    """
    [Continuous]
    2 ALT Pressure Altitude [8, -1,500~80,000ft]= "PRESALT"
    5 V(M) Calibrated Air Speed(CAS) [8, 70~1,000kts]= "CAS"
    35 RPM N2(Fan Speed) [8, 0~120%]= "N2"
    36 RPM N1(Core Speed) [8, 0~120%]= "N1"
    37 PLA Power Lever Angle [8, 0~140]= "PLA"
    38 EGT(T_t6) Turbine Exit Temperature [8, 0~1,200℃]= "TET"
    39 T1 Fan Inlet Temperature(T1) [8, -80~170℃]= "T1"
    40 VEN Ven Actuator Stroke [8, 0~100%]= "VENACTSTKE"
    46 T0 True Free Air Temperature(TFAT) [1, 173~323°K]= "TFAT"
    61 P0 Selected Static Pressure [8, 0.8~32(in-Hg)]= "STATPRES"
    68 FVG Fan Variable Geometry [8, -10~100°(deg)]= "FVG"
    69 CVG Core Variable Geometry [8, -10~100°(deg)]= "CVG"
    """
    
    # 데이터 전처리 수행
    df = process_data(file_path, x_name, y_names)

    # 원시 데이터를 csv 파일로 저장
    print("Saving raw data dictionary as csv:")
    output_to_csv(df, file_path, 'processed')
    
 
# 메인 함수 호출
if __name__ == "__main__":
    main()
    