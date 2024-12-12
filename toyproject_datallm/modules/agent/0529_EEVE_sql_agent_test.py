import os
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import sqlite3
from langchain.chains.sql_database.query import create_sql_query_chain
from langchain_community.utilities import SQLDatabase
from langchain.prompts import ChatPromptTemplate

# GPU 사용 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 중인 장치: {device}")

# Helper 함수
def get_schema(_):
    return db.get_table_info()

def run_query(query):
    return db.run(query)

# 데이터베이스 경로
db_path = "/workspace/youngwoo/toyproject-datallm/modules/agent/tmp_dataset/test_database.db"

# 데이터베이스 연결 초기화
db = SQLDatabase.from_uri(f"sqlite:///{db_path}")

# 모델 로드
model_id = "yanolja/EEVE-Korean-Instruct-10.8B-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto").to(device)

# SQL 쿼리 생성을 위한 프롬프트
template_query = """
사용자의 질문에 답변하기 위해 데이터를 로드하는 SQL문을 생성하세요.
테이블 스키마를 바탕으로 반드시 데이터를 불러오기 위한 SQL 문법만 (SELECT, WHERE, FROM, BETWEEN 등) 사용하세요.

table 이름은 test_table 입니다.
{schema}

Question: {question}
SQL Query: """

prompt_query = ChatPromptTemplate.from_template(template_query)

# 자연어 질문으로부터 SQL 쿼리 생성 함수
def generate_sql_query(question):
    # 스키마 정보 가져오기
    schema = get_schema(None)

    # 프롬프트 준비
    prompt = prompt_query.format(schema=schema, question=question)

    # 챗 템플릿을 위한 메시지 생성
    messages = [
        {"role": "system", "content": "당신은 아주 유능한 데이터베이스 전문가입니다."},
        {"role": "user", "content": prompt}
    ]

    # 프롬프트 토크나이즈
    input_ids = tokenizer(prompt, return_tensors="pt").to(device)

    # 종료 토큰 정의
    terminators = [tokenizer.eos_token_id]

    # 응답 생성
    outputs = model.generate(
        input_ids.input_ids,
        max_new_tokens=1000
    )

    # 응답 디코드
    output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    print(output_text)


    response = outputs[0][input_ids.input_ids.shape[-1]:]
    sql_query = tokenizer.decode(response, skip_special_tokens=True).strip()

    # 응답에서 SQL 쿼리 추출
    sql_query = sql_query.split("SQL Query:")[1].strip() if "SQL Query:" in sql_query else sql_query
    return sql_query

def main():
    question = "3월 24일 데이터들에 대해 분 단위로 변환하여 기초 통계 정보를 알려주세요. (최소, 최대, 평균)"

    # 자연어 질문으로부터 SQL 쿼리 생성
    sql_query = generate_sql_query(question)
    #print("Generated SQL Query:", sql_query)

if __name__ == "__main__":
    main()
