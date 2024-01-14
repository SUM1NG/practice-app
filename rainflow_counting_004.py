import csv
import os
import matplotlib.pyplot as plt

def remove_special_characters(file_path):
    # 특수 문자 제거
    file_path = file_path.replace("\"", "").replace("'", "")
    return file_path

def convert_to_csv(file_path):
    # 파일이 csv 형식이 아니라면 csv 파일로 변환
    if not file_path.lower().endswith('.csv'):
        csv_file_path = os.path.splitext(file_path)[0] + '.csv'
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        with open(csv_file_path, 'w', encoding='utf-8') as csv_file:
            csv_file.write(content)
        return csv_file_path
    return file_path

def read_csv_file(file_path):
    # csv 파일 읽기
    data = []
    with open(file_path, 'r', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            data.append(row)
    return data

def remove_columns(data, x_name, y_names):
    # x_name과 y_names에 해당하지 않는 열 제거
    x_index = -1
    y_indices = []
    new_data = []
    for i, row in enumerate(data):
        if i == 0:
            for j, column in enumerate(row):
                if column == x_name:
                    x_index = j
                elif column in y_names:
                    y_indices.append(j)
            new_data.append(row)
        elif i == 1:
            new_data.append(row)
            new_data.append([''] * len(row))
        else:
            new_row = [row[x_index]]
            for y_index in y_indices:
                new_row.append(row[y_index])
            new_data.append(new_row)
    return new_data

def process_repeated_values(data):
    # 중복된 값 처리
    for j in range(1, len(data[0])):
        column_values = [row[j] for row in data[2:] if len(row) > j and row[j] != '']
        if len(column_values) > 2 and len(set(column_values)) == 1 and column_values[0] != '0':
            min_value = min(column_values)
            for i in range(2, len(data)):
                if len(data[i]) > j and data[i][j] == column_values[0]:
                    data[i][j] = min_value
    return data

def process_blank_rows(data):
    # 빈 행 처리
    for j in range(1, len(data[0])):
        for i in range(2, len(data)-1):
            if data[i][j] == '':
                prev_value = data[i-1][j] if len(data[i-1]) > j else ''
                next_value = data[i+1][j] if len(data[i+1]) > j else ''
                if prev_value != '' and next_value != '':
                    data[i][j] = (float(prev_value) + float(next_value)) / 2
                elif prev_value == '' and next_value != '':
                    data[i][j] = next_value
                elif prev_value != '' and next_value == '':
                    data[i][j] = prev_value
    return data

def create_processed_csv(data, file_path):
    # "[processed]" 접두어를 추가하여 새로운 csv 파일 생성
    processed_file_path = os.path.splitext(file_path)[0] + "[processed].csv"
    with open(processed_file_path, 'w', encoding='utf-8', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerows(data)
    return processed_file_path

def create_data_dict(data, x_name, y_names):
    # 데이터 딕셔너리 생성
    data_dict = {
        'x_axis': x_name,
        'y_axis': y_names,
        'units': data[1][1:],
        'data': []
    }
    for i in range(2, len(data)):
        data_dict['data'].append(data[i][1:])
    return data_dict

def draw_graph(data_dict):
    # 그래프 그리기
    x_axis = data_dict['x_axis']
    y_axis = data_dict['y_axis']
    data = data_dict['data']

    plt.figure(figsize=(8, 4))
    for i, y_name in enumerate(y_axis):
        plt.plot(data[0], data[i+1], label=y_name)

    # 그래프 설정
    plt.xlabel(x_axis)
    plt.ylabel(','.join(y_axis))
    plt.title('Graph')
    plt.legend(loc='upper right')
    plt.xticks(rotation=90)
    plt.yticks([i for i in range(-15, 16)])

    # 음수 값이 있는 경우 X 축 설정
    if any([any([float(value) < 0 for value in row[1:]]) for row in data]):
        plt.xticks([i for i in range(-20, 21)])

    # 그래프 표시
    plt.show()

def main():
    # 사용자로부터 파일 경로 입력 받기
    file_path = input("파일 경로를 입력하세요: ")

    # 특수 문자 제거
    file_path = remove_special_characters(file_path)
    print(f"특수 문자가 제거되었습니다: {file_path}")

    # csv 파일로 변환
    file_path = convert_to_csv(file_path)
    print(f"csv 파일로 변환되었습니다: {file_path}")

    # csv 파일 읽기
    data = read_csv_file(file_path)
    print(f"csv 파일이 읽혔습니다: {file_path}")

    # x_name과 y_names 입력 받기
    x_name = input("x축의 이름을 입력하세요: ")
    y_names = input("y축의 이름을 쉼표로 구분하여 입력하세요: ").split(',')

    # 열 제거
    data = remove_columns(data, x_name, y_names)
    print(f"주어진 열만 남겼습니다.")

    # 중복된 값 처리
    data = process_repeated_values(data)
    print(f"중복된 값이 처리되었습니다.")

    # 빈 행 처리
    data = process_blank_rows(data)
    print(f"빈 행이 처리되었습니다.")

    # 새로운 csv 파일 생성
    processed_file_path = create_processed_csv(data, file_path)
    print(f"새로운 csv 파일이 생성되었습니다: {processed_file_path}")

    # 데이터 딕셔너리 생성
    data_dict = create_data_dict(data, x_name, y_names)

    # 그래프 그리기
    draw_graph(data_dict)

if __name__ == "__main__":
    main()
