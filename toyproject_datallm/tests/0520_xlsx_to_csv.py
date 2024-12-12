import pandas as pd

# # Load the .xlsm file
# file_path = '/workspace/youngwoo/toyproject-datallm/dataset.xlsm'
# xlsm_file = pd.ExcelFile(file_path)

# # Display sheet names
# print(xlsm_file.sheet_names)

# # Load the first sheet into a DataFrame
# df = pd.read_excel(xlsm_file, sheet_name=xlsm_file.sheet_names[0])

# # Save the DataFrame to a CSV file
# df.to_csv('/workspace/youngwoo/toyproject-datallm/dataset.csv', index=False)

# print("Conversion to CSV completed successfully.")

# csv_file = '/workspace/youngwoo/toyproject-datallm/dataset.csv'

# # 첫 번째 행을 열 이름으로 설정하여 엑셀 파일 읽기
# df = pd.read_csv(csv_file, header=2)
# df.rename(columns={'Unnamed: 0': 'datetime'}, inplace=True)
# # 수정된 데이터프레임 확인
# print("수정된 데이터프레임:")
# print(df.head())

# # 수정된 데이터프레임을 새로운 CSV 파일로 저장
# new_csv_file = '/workspace/youngwoo/toyproject-datallm/dataset_modified.csv'
# df.to_csv(new_csv_file, index=False)

# print(f"수정된 데이터프레임이 '{new_csv_file}' 파일에 저장되었습니다.")

# 엑셀 파일 경로
excel_file = '/workspace/youngwoo/toyproject-datallm/dataset.xlsm'

# 엑셀 파일의 모든 시트를 데이터프레임으로 읽어오기
all_sheets = pd.read_excel(excel_file, sheet_name=None, header=2)

# 모든 시트를 하나의 데이터프레임으로 합치기
df_list = []
for sheet_name, df in all_sheets.items():
    df.rename(columns={'Unnamed: 0': 'datetime'}, inplace=True)
    df_list.append(df)

# 하나의 데이터프레임으로 병합
merged_df = pd.concat(df_list, ignore_index=True)

# 수정된 데이터프레임 확인
print("수정된 데이터프레임:")
print(merged_df.head())

# 수정된 데이터프레임을 새로운 CSV 파일로 저장
new_csv_file = '/workspace/youngwoo/toyproject-datallm/tests/all_data.csv'
merged_df.to_csv(new_csv_file, index=False)

print(f"수정된 데이터프레임이 '{new_csv_file}' 파일에 저장되었습니다.")