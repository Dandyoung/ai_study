import sqlite3
import pandas as pd
import os
from openai import OpenAI
import re

# OpenAI 클라이언트 초기화
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


file_path = '/workspace/youngwoo/toyproject-datallm/test_dataset.db'
conn = sqlite3.connect(file_path)
query = "SELECT * FROM test_table"
df = pd.read_sql_query(query, conn)

def _get_table_info(df):
    return df.head(3)

def generate_sql_query(question, table_info, table_name):
    prompt = f"""
    주어진 질문을 SQLite 쿼리로 변환해주세요.
    다음 형식을 사용하세요:

    아래는 SQL 쿼리 변환을 위한 {table_name} 테이블의 열, 행 예시입니다 : 
    {table_info}.

    변환할 질문: {question}
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

question = "해당 데이터베이스에는 몇개의 열이 있어?"
table_info = _get_table_info(df)
generate_sql_query(question=question, table_info=table_info, table_name='test_table')


