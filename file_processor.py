# 필요한 라이브러리 임포트
import os

# 필요한 패키지들이 설치되어 있는지 확인하고, 없다면 설치
try:
    import pandas as pd
except ImportError:
    os.system('pip install pandas')

# 데이터 처리하는 함수
def process_data(file_path, x_name, y_names, y_name, sampling_rate, multiply_constant):
    # 데이터 읽기
    df = pd.read_csv(file_path)  # csv 파일 읽어와서 DataFrame 생성

    # 단위 제거
    df = df.drop(df.index[0])
    
    # 불필요한 열 삭제 및 재배치
    cols = [col for col in [x_name] + y_names if col in df.columns]  # DataFrame에 존재하는 열만 선택
    if len(cols) < len([x_name] + y_names):
        print("Some columns are not found in the DataFrame.")
    df = df[cols]  # 필요한 열만 선택해서 DataFrame 재생성
    
    # 샘플링주기에 따라 데이터 삭제
    #df = adjust_sampling_rate(df, sampling_rate)
    
    # y_names 열에서 결측치 또는 빈칸 삭제
    df = df.dropna(subset=y_names)
    for col in y_names:
        df = df[df[col].notnull()]
    df = df[df[y_name].notnull()]

    # 숫자를 float으로 변경
    df[y_names] = df[y_names].astype(float)
    
    # 각 열의 소수점 이하 최대 자릿수를 찾음
    decimal_places = find_decimal_places(df, y_names)
    
    # 각 열을 상수로 곱하고, 원래의 소수점 이하 자릿수로 반올림
    for col in y_names:
        df[col] = (df[col] * multiply_constant).round(decimal_places[col])

    # "TIME" 열의 값을 1부터 시작하는 연속된 정수로 바꾸기
    df[x_name] = range(1, len(df) + 1)

    # 열 이름 변경
    df = rename_columns(df, y_names)
    
    return df

# 각 열의 소수점 이하 최대 자릿수를 찾는 함수
def find_decimal_places(df, cols):
    decimal_places = {}
    for col in cols:
        # 각 값이 소수인지 확인
        is_decimal = df[col].apply(lambda x: '.' in str(x))
        # 소수라면 소수점 이하 자릿수를 찾음
        if is_decimal.any():
            places = df[col].apply(lambda x: len(str(x).split('.')[1]) if '.' in str(x) else 0)
            decimal_places[col] = places.max()
        else:
            decimal_places[col] = 0
    return decimal_places

def rename_columns(df, y_names):
    # 열 이름을 변경할 딕셔너리 생성
    rename_dict = {}
    for i, old_name in enumerate(y_names):
        # Create the new name
        new_name = 'var_' + str(i+1).zfill(2)
        
        # Add the old name and new name to the dictionary
        rename_dict[old_name] = new_name

    # 열 이름 변경
    df = df.rename(columns=rename_dict)
    
    return df    

def adjust_sampling_rate(df, sampling_rate):
    
    # 샘플링 레이트에 따라 열 삭제
    if sampling_rate == 16:
        pass  # 원본 그대로 유지
    elif sampling_rate == 8:
        df = df.iloc[::2]  # 매 두 번째 행만 유지
    elif sampling_rate == 4:
        df = df.iloc[::4]  # 매 네 번째 행만 유지
    elif sampling_rate == 1:
        df = df.iloc[::16]  # 매 16번째 행만 유지
    else:
        print("Invalid sampling rate. Please choose from 1, 4, 8, or 16.")
        return None
    
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
  
    file_path = "C:/Users/APTech-dev03/Documents/2. 프로젝트/Acts Tech 제공할 자료/50-004_230414/50-004_2304141.csv"
    #file_path = "C:/Users/APTech-dev03/Documents/2. 프로젝트/Acts Tech 제공할 자료/50-004_230417/50-004_2304171.csv"
    
    #file_path = "C:/Users/Paul Kim/Documents/@_DOCS_@/@@APTECH/FA-50 PHMS/VADR/50-004_230414/50-004_2304141.csv"
    x_name = "TIME"  # x축의 이름을 "TIME"으로 설정
    y_names = ["PRESALT", "CAS", "N2", "N1", "PLA", "TET", "T1", "VENACTSTKE", "TFAT", "STATPRES", "FVG", "CVG"]  # y축 이름 설정

    # 데이터 전처리 수행
    df = process_data(file_path, x_name, y_names, "TFAT", 8, 0.927392981538929836)

    # 원시 데이터를 csv 파일로 저장
    print("Saving raw data dictionary as csv:")
    output_to_csv(df, file_path, 'processed')
    
 
# 메인 함수 호출
if __name__ == "__main__":
    main()
    



"""
# 사용자로부터 필요한 정보 입력받기
def get_user_input():
    file_path = input("Enter the file path: ").replace('"', '')  # 파일 경로 입력 받음
    x_name = input("Enter the x-axis name: ").strip()  # x축 이름 입력 받음
    y_names = [name.strip() for name in input("Enter the y-axis names (separated by comma): ").split(',')]  # y축 이름들 입력 받음
    return file_path, x_name, y_names  # 입력 받은 정보 반환


    # 초 단위 데이터만 남기고 제거
    #df = df[df[x_name].str.endswith('.000')]

    # 빈 값은 앞뒤 값의 평균으로 대체
    #for y_name in y_names:  # 각 y_name에 대해
    #    df[y_name].interpolate(method='linear', limit_direction ='both', inplace=True)  # 빈 값이 있는 경우, 앞뒤 값의 선형적인 평균으로 대체
"""


"""
    [Continuous]
    [var_01] 2 ALT Pressure Altitude [8, -1,500~80,000ft]= "PRESALT"
    [var_02] 5 V(M) Calibrated Air Speed(CAS) [8, 70~1,000kts]= "CAS"
    [var_03] 35 RPM N2(Fan Speed) [8, 0~120%]= "N2"
    [var_04] 36 RPM N1(Core Speed) [8, 0~120%]= "N1"
    [var_05] 37 PLA Power Lever Angle [8, 0~140]= "PLA"
    [var_06] 38 EGT(T_t6) Turbine Exit Temperature [8, 0~1,200℃]= "TET"
    [var_07] 39 T1 Fan Inlet Temperature(T1) [8, -80~170℃]= "T1"
    [var_08] 40 VEN Ven Actuator Stroke [8, 0~100%]= "VENACTSTKE"
    [var_09] 46 T0 True Free Air Temperature(TFAT) [1, 173~323°K]= "TFAT"
    [var_10] 61 P0 Selected Static Pressure [8, 0.8~32(in-Hg)]= "STATPRES"
    [var_11] 68 FVG Fan Variable Geometry [8, -10~100°(deg)]= "FVG"
    [var_12] 69 CVG Core Variable Geometry [8, -10~100°(deg)]= "CVG"
"""