import sqlite3
import pandas as pd
import os
from openai import OpenAI
import os
import sqlite3
import pandas as pd

# OpenAI API 키 설정
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
def generate_query(question):
    prompt = f"""
    주어진 텍스트를 df(pandas 데이터 프레임)에서 검색할 수 있는 python 코드로 바꿔주세요.
    추가적인 설명이나 주석 없이, 코드 블록도 없이 코드만 반환해 주세요.
    마지막 줄에 반드시 'result' 변수에 결과를 저장해주세요.
    """
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "당신은 아주 유능한 coder입니다."},
            {"role": "user", "content": prompt + '\n' + question}
        ],
        max_tokens=150
    )
    return completion.choices[0].message.content.strip()
def preprocess_code(code):
    # ``` 블록 제거
    code = code.replace('```python', '').replace('```', '').strip()
    return code

def execute_generated_code(code, df):
    globals_dict = {'pd': pd, 'df': df}
    try:
        code = preprocess_code(code)
        exec(code, globals_dict)
        result = globals_dict.get('result', 'No result variable found.')
        return result
    except Exception as e:
        return str(e)

# 상대 경로 설정
relative_path = '/workspace/youngwoo/toyproject-datallm/test_dataset.db' 

# 현재 작업 디렉토리 확인
print("현재 작업 디렉토리:", os.getcwd())

# 절대 경로 얻기
file_path = os.path.abspath(relative_path)
print("데이터베이스 파일 절대 경로:", file_path)

# 파일 존재 여부 확인
if not os.path.exists(file_path):
    print(f"파일이 존재하지 않습니다: {file_path}")
else:
    try:
        # SQLite 데이터베이스 연결
        conn = sqlite3.connect(file_path)

        # 데이터프레임으로 변환
        query = "SELECT * FROM test_table"
        df = pd.read_sql_query(query, conn)
        print("기존 데이터프레임:")
        print(df.head())
        # 첫 번째 행을 컬럼 이름으로 사용
        df.columns = df.iloc[0]

        # 첫 번째 행 삭제
        df = df.drop(df.index[0])

        # 필요 없는 Unnamed 열 삭제
        df = df.loc[:, ~df.columns.astype(str).str.startswith('Unnamed')]

        # 인덱스를 재설정
        df.reset_index(drop=True, inplace=True)
        # 데이터프레임 확인
        print("수정된 데이터프레임:")
        print(df.head())
    except sqlite3.OperationalError as e:
        print(f"데이터베이스 파일을 열 수 없습니다: {e}")

# question = "3월 24일 데이터에 대해 분 단위로 변환하여 기초 통계 정보를 알려주세요. (최소, 최대, 평균)"
# generated_code = generate_query(question=question)
# print(f"Generated Code:\n{generated_code}")

# # Generated Code 실행
# result = execute_generated_code(generated_code, df)
# print(f"Result: {result}")