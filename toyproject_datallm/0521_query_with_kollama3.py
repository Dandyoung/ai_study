
import sqlite3
import pandas as pd
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sqlalchemy import create_engine
import json, re
import transformers
import torch
# SQLite 데이터베이스 파일 경로 설정
file_path = '/workspace/youngwoo/toyproject-datallm/test_dataset.db'
database_url = f'sqlite:///{file_path}'

# SQLAlchemy 데이터베이스 엔진 생성
engine = create_engine(database_url)

# 로컬 LLaMA 모델 경로 설정
local_model_path = '/workspace/huggingface/Ko-Llama3-Luxia-8B'

# 토크나이저와 모델 로드
tokenizer = AutoTokenizer.from_pretrained(local_model_path)
model = AutoModelForCausalLM.from_pretrained(local_model_path, torch_dtype=torch.bfloat16)

# CUDA 장치가 있는 경우 GPU로 모델을 이동
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 텍스트 생성 파이프라인 설정
text_generation_pipeline = pipeline("text-generation",
                                    model=model, 
                                    tokenizer=tokenizer, 
                                    device=0 if torch.cuda.is_available() else -1)


pipeline = transformers.pipeline(
    "text-generation", model=model, tokenizer=tokenizer, max_length=1000, model_kwargs={"torch_dtype": torch.bfloat16}, device=0 if torch.cuda.is_available() else -1
)


# 데이터베이스 연결 및 데이터프레임 생성
conn = sqlite3.connect(file_path)
query = "SELECT * FROM test_table"
df = pd.read_sql_query(query, conn)

prompt = f"""<|begin_of_text|>
주어진 질문을 SQL query로 변환해주세요.
반복적인 메시지나 문구를 사용하지마세요.

참고 자료 : 
{df.head(1).to_string()}
질문 : 전체 기간에 Null 데이터가 있나요?
"""

response = pipeline(prompt, do_sample=True, temperature=1.0, top_p=0.9)
response_text = response[0]['generated_text']

print(response_text)


# # 테이블 정보 가져오기
# def get_table_info(df):
#     return df.head(3).to_string()

# # SQL 쿼리 생성 함수
# def generate_sql_query(question, table_info, table_name):
#     prompt = f"""<|begin_of_text|>
#     주어진 질문을 SQLite 쿼리로 변환해주세요.
#     반복적인 메시지나 문구를 사용하지마세요.
#     추가적인 설명이나 주석 없이 SQL 쿼리만 작성해주세요.
    
#     아래는 SQL 쿼리 변환을 위한 {table_name} 테이블의 열, 행 예시입니다 : 
#     {table_info}.

#     변환할 질문: {question}
#     """
#     # terminators = [
#     # tokenizer.eos_token_id,
#     # tokenizer.convert_tokens_to_ids("<|eot_id|>")
#     # ]

#     # 텍스트 생성 파이프라인을 사용하여 응답 생성
#     # pad_token_id를 eos_token_id로 설정
#     response = text_generation_pipeline(prompt, max_length=1000)

#     response_text = response[0]['generated_text']
    
#     # 응답 내용 출력 (디버깅용)
#     # print("실제 출력:", response_text)
    
#     return response_text

# # 데이터베이스 연결 및 데이터프레임 생성
# conn = sqlite3.connect(file_path)
# query = "SELECT * FROM test_table"
# df = pd.read_sql_query(query, conn)

# # 테이블 정보 및 질문 설정
# table_info = get_table_info(df)
# question = "<|begin_of_text|>해당 데이터베이스에는 몇개의 열이 있어?\n"
# # question = "3월 24일 데이터에 대해 분 단위로 변환하여 기초통계 정보를 알려주세요.(최소, 최대, 평균)"

# # SQL 쿼리 생성
# sql_query = generate_sql_query(question=question, table_info=table_info, table_name='test_table')
# # SQL 쿼리에서 ```으로 감싸진 부분 추
# # 결과를 JSON 형식으로 저장
# print(sql_query)
# output = {"response": sql_query}
# output_file_path = '/workspace/youngwoo/toyproject-datallm/output.json'

# with open(output_file_path, 'w', encoding='utf-8') as f:
#     json.dump(output, f, ensure_ascii=False, indent=4)

# print(f"SQL Query has been saved to {output_file_path}")