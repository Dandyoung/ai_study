
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
local_model_path = '/workspace/huggingface/Llama-3-Open-Ko-8B-Instruct-preview'

# 토크나이저와 모델 로드
tokenizer = AutoTokenizer.from_pretrained(local_model_path)
model = AutoModelForCausalLM.from_pretrained(local_model_path, torch_dtype=torch.bfloat16)

# CUDA 장치가 있는 경우 GPU로 모델을 이동
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


messages = [
    # 이 system pormpt는 바꾸면 안되나보다..이게 학습에 계속 사용됬나보다.. 이거 바꾸니까 출력이 너무 이상함..
    {"role": "system", "content": "친절한 챗봇으로서 상대방의 요청에 최대한 자세하고 친절하게 답하자. 모든 대답은 한국어(Korean)으로 대답해줘."},
    {"role": "user", "content": "전체 기간에 Null 데이터가 있는지 확인하는 sql query문 말해줘."},
]

input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = model.generate(
    input_ids,
    max_new_tokens=1000,
    eos_token_id=terminators,
    do_sample=True,
    temperature=1,
    top_p=0.9,
)
response = outputs[0][input_ids.shape[-1]:]
print(tokenizer.decode(response, skip_special_tokens=True))