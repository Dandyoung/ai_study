import sqlite3
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sqlalchemy import create_engine
import torch
from langchain_experimental.agents import create_pandas_dataframe_agent
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.llms.base import LLM

class CustomLlamaForCausalLM(LLM):
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def _call(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    @property
    def _identifying_params(self):
        return {"name": "CustomLlamaForCausalLM"}

    @property
    def _llm_type(self) -> str:
        return "custom_llama_for_causal_lm"

# 로컬 LLaMA 모델 경로 설정
local_model_path = '/workspace/huggingface/Ko-Llama3-Luxia-8B'

# 토크나이저와 모델 로드
tokenizer = AutoTokenizer.from_pretrained(local_model_path)
model = AutoModelForCausalLM.from_pretrained(local_model_path, torch_dtype=torch.bfloat16)

# CUDA 장치가 있는 경우 GPU로 모델을 이동
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# CustomLlamaForCausalLM 인스턴스 생성
custom_llm = CustomLlamaForCausalLM(model, tokenizer, device)

# SQLite 데이터베이스에서 데이터 로드
file_path = '/workspace/youngwoo/toyproject-datallm/test_dataset.db'
conn = sqlite3.connect(file_path)
query = "SELECT * FROM test_table"
df = pd.read_sql_query(query, conn)

# LangChain create_pandas_dataframe_agent 함수에 custom_llm을 전달합니다
agent = create_pandas_dataframe_agent(custom_llm, df, prefix="당신은 훌륭한 데이터베이스 도우미입니다.", system_message="당신은 사용자 질문을 SQL query문으로 변환해야합니다.")

response = agent.invoke("해당 데이터베이스에는 몇개의 열이 있어?\n")
print(response)
