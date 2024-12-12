import sqlite3
import pandas as pd
import os
from openai import OpenAI
import re

# OpenAI 클라이언트 초기화
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# 상대 경로 설정
relative_path = '/workspace/youngwoo/toyproject-datallm/test_dataset.db'  # Jupyter 노트북 파일의 상대 경로로 변경

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
        query = "SELECT * FROM test_table"  # 실제 테이블 이름으로 변경
        df = pd.read_sql_query(query, conn)

        # 첫 번째 행을 새로운 열 이름으로 설정하고, 두 번째 행을 무시
        df.columns = df.iloc[0]
        df = df[1:]

        # 첫 번째 열을 'datetime'으로 변경
        df.iloc[0, 0] = 'datetime'
        df.columns = df.iloc[0]
        df = df[1:]

        # 인덱스를 재설정
        df.reset_index(drop=True, inplace=True)

        # 데이터프레임 확인
        print("수정된 데이터프레임:")
        print(df.head())
    except sqlite3.OperationalError as e:
        print(f"데이터베이스 파일을 열 수 없습니다: {e}")

# 테이블 정보 가져오기
def get_table_info(df, table_name):
    return {table_name: df.columns.tolist()}

# 사용자 질문에 따라 SQL 쿼리 생성
def generate_sql_query(question, table_info):
    prompt = f"""
    주어진 질문을 바탕으로, 먼저 실행할 구문적으로 올바른 SQLite 쿼리를 작성하고, 그런 다음 쿼리 결과를 확인하여 답을 출력하세요.
    다음 형식을 사용하세요:

    Question: "질문 내용"
    SQLQuery: "실행할 SQL 쿼리"
    SQLResult: "SQL 쿼리 결과"
    Answer: "최종 답변"

    다음 테이블만 사용하세요:
    {table_info}.

    Question: {question}
    """
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "당신은 아주 유능한 데이터베이스 전문가입니다."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150
    )
    response_text = completion.choices[0].message.content
    
    # 응답 내용 출력 (디버깅용)
    print("Response text from OpenAI:", response_text)
    
    # 정규식을 사용하여 SQL 쿼리 추출
    match = re.search(r'SQLQuery:\s*"(.*?)"', response_text)
    if match:
        sql_query = match.group(1)
        return sql_query
    else:
        raise ValueError("SQL query not found in the response")

# SQL 쿼리 실행 및 결과 반환
def execute_sql_query(df, sql_query, table_name):
    try:
        # PRAGMA table_info 쿼리 처리를 위해 직접 Pandas 처리
        if sql_query.lower().startswith("pragma table_info"):
            # 컬럼 정보를 반환
            columns = df.columns.tolist()
            num_columns = len(columns)
            result = f"총 {num_columns}개의 열이 있습니다."
            return result
        else:
            # SQL 쿼리를 Pandas의 코드로 변환
            query = sql_query.replace(";", "")
            # NULL 값 확인 쿼리
            if "null" in sql_query.lower():
                null_count = df.isnull().sum().sum()
                result = f"해당 DB에는 총 {null_count}개의 NULL 값이 있습니다."
                return result
            # 데이터프레임의 열 이름을 기반으로 쿼리를 Pandas의 eval()로 실행
            result_df = df.query(query)
            return result_df
    except Exception as e:
        return str(e)

# 체인 실행 함수
def run_chain(question, df, table_name):
    table_info = get_table_info(df, table_name)
    table_info_str = "\n".join([f"{table}: {', '.join(columns)}" for table, columns in table_info.items()])
    sql_query = generate_sql_query(question, table_info_str)
    print(f"Generated SQL Query: {sql_query}")
    result = execute_sql_query(df, sql_query, table_name)
    return result

# 예제 실행
question = "해당 DB에는 NULL값이 있어 ?"
table_name = "test_table"
result = run_chain(question, df, table_name)
print(f"Result: {result}")
