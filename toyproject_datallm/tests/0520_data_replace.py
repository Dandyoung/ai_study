import pandas as pd
import sqlite3

excel_file = '/workspace/youngwoo/toyproject-datallm/dataset.xlsm'
sqlite_db = '/workspace/youngwoo/toyproject-datallm/test_dataset.db'

# 첫 번째 행을 열 이름으로 설정하여 엑셀 파일 읽기
df = pd.read_excel(excel_file, header=2, sheet_name=0)
df.rename(columns={'Unnamed: 0': 'datetime'}, inplace=True)
# 수정된 데이터프레임 확인
print("수정된 데이터프레임:")
print(df.head())

# SQLite 데이터베이스 연결 (없으면 생성)
conn = sqlite3.connect(sqlite_db)
cursor = conn.cursor()

# 데이터프레임을 SQLite로 내보내기 (테이블 이름은 'my_table'로 설정)
df.to_sql('test_table', conn, if_exists='replace', index=False)

# 연결 닫기
conn.close()
print("엑셀 파일이 SQLite 데이터베이스로 성공적으로 변환되었습니다.")
