# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
# from langchain_community.llms import HuggingFacePipeline
# from langchain.prompts import PromptTemplate
# from langchain.chains import LLMChain
# from langchain_core.output_parsers import StrOutputParser

# # 로컬 경로 설정
# local_model_path = "/workspace/huggingface/Meta-Llama-3-8B-Instruct"

# # 토크나이저와 모델 로드
# tokenizer = AutoTokenizer.from_pretrained(local_model_path)
# model = AutoModelForCausalLM.from_pretrained(local_model_path, torch_dtype=torch.bfloat16)

# # CUDA 장치가 있는 경우 GPU로 모델을 이동
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# # 텍스트 생성 파이프라인 설정
# text_generation_pipeline = pipeline("text-generation",
#                                     model=model, 
#                                     tokenizer=tokenizer, 
#                                     device=0 if torch.cuda.is_available() else -1,
#                                     max_length=200,  # 출력 길이 제한
#                                     num_return_sequences=1,  # 생성할 응답 수
#                                     do_sample=True,  # 샘플링 사용
#                                     top_k=1,  # top-k 샘플링
#                                     top_p=0.95)  # top-p 샘플링

# # HuggingFace LLM 설정
# llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

# # 프롬프트 템플릿 설정
# prompt = PromptTemplate(
#     input_variables=["input"],
#     template=
#     """
#     You are a very skilled coder.
#     Please convert the given text into SQLite.
#     Return only the code without any additional explanations, comments, or code blocks.
#     {input}
#     """
# )

# # 체인 구성
# chain = LLMChain(prompt=prompt, llm=llm, output_parser=StrOutputParser())

# # 체인 실행: 'input' 변수 사용
# message = chain.invoke({"input": "3월 24일 데이터에 대해 분 단위로 변환하여 기초 통계 정보를 알려주세요. (최소, 최대, 평균)"})

# print(f"message1: {message['text']}")  # 따옴표를 이스케이프 처리하여 중괄호 내부를 올바르게 인식하도록 함


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain_openai import ChatOpenAI
from langchain.chains.sql_database.query import create_sql_query_chain
from langchain_core.prompts import PromptTemplate
from langchain_community.utilities import SQLDatabase

# 데이터베이스 설정
db = SQLDatabase.from_uri("sqlite:////workspace/youngwoo/toyproject-datallm/test_dataset.db")

# 로컬 경로 설정
local_model_path = "/workspace/huggingface/Meta-Llama-3-8B-Instruct"

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
                                    device=0 if torch.cuda.is_available() else -1,)

# HuggingFace LLM 설정
llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

chain = create_sql_query_chain(llm, db)
response = chain.invoke({"question": "해당 db에는 몇개의 열이 있어?"})
print(response)