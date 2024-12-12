from ml_collections import ConfigDict
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import sqlite3
import pandas as pd
from transformers import AutoTokenizer
from langchain_core.runnables import RunnableLambda
from loguru import logger
import subprocess
import re
from langchain.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain_community.chat_models.huggingface import ChatHuggingFace

csv_file_path = 'data.csv'
sqlite_file_path = 'data.db'
# 이부분 한번에 줄 수 있는지 생각
table_name = 'test_table'
df = pd.read_csv(csv_file_path)
conn = sqlite3.connect(sqlite_file_path)
cursor = conn.cursor()
df.to_sql(table_name, conn, if_exists='replace', index=False)
conn.close()

db_path = "data.db"
db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
schema = db.get_table_info()

# print(schema)

llm = ChatOpenAI(temperature=0, model_name='gpt-4')
TEMPLATE_MODEL = 'gpt'

def run_query(query):
    return db.run(query)


# execute_sql_query 함수는 주어진 데이터베이스 경로(db_path)와 SQL 쿼리(query)를 사용하여 데이터베이스에 접속하고, 쿼리를 실행한 후 결과를 반환
def execute_sql_query(db_path, query):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(query)
    results = cursor.fetchall()
    columns = [description[0] for description in cursor.description]
    conn.close()
    return columns, results


# execute_sql_query 함수는 주어진 데이터베이스 경로(db_path)와 SQL 쿼리(query)를 사용하여 데이터베이스에 접속하고, 쿼리를 실행한 후 결과를 반환
def execute_sql_query_to_df(db_path, query):
    # 데이터베이스 연결 및 쿼리 실행
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(query)
    results = cursor.fetchall()
    columns = [description[0] for description in cursor.description]
    conn.close()
    
    # 결과를 데이터프레임으로 변환
    df = pd.DataFrame(results, columns=columns)
    return df


def get_input_gpt(user_message):
    return user_message

def get_template(template,var="custom"):
    if var == "gpt":
        return get_input_gpt(template)

TemplateConfig = ConfigDict()

TemplateConfig.make_query_split = """
사용자 질문 : {query}
사용자의 질문을 두가지로 분리하십시오.
1. 데이터베이스에서 데이터를 가져오라는 요청문
2. 사용자의 요구사항

예시 : 1월12일자 데이터의 컬럼별 평균값을 구해줘
요청1 : 1월12일자 데이터를 가져오세요.
요청2 : 컬럼별 평균값을 구하세요.

요청1 : ...
요청2 : ...
"""

TemplateConfig.make_sql_query = """
사용자의 질문에 답변하기 위해 데이터베이스에서 데이터를 가져와야합니다.
테이블 스키마를 바탕으로 데이터를 가져오도록 SQL문을 생성하세요.
무조건 SQL문 만을 생성하세요.

table 이름은 test_table 입니다.
{schema}
 
Question: {query}
출력 양식 : ```sql\n<sql code>\n```"""



TemplateConfig.make_code = """
당신은 python을 이용한 데이터분석 전문가입니다.
분석에 사용될 데이터(data.csv)는 가지고있습니다.
사용자 질문에 답하기 위해, 분석에 필요한 python code를 작성해주세요.
무조건 마지막에 사용자 질문에 대한 답변을 (result.csv)로 저장하는 python code를 작성하세요.
무조건 python code만 생성하세요.
Table Schema : {schema}
Question: {query}
Column Information : {columns}
Data Row 1 : {sample}
출력 양식 : ```python\n<python code>\n```"""


TemplateConfig.select_visualization= """
아래 데이터는 '{query}'에 따른 데이터를 데이터베이스에서 가져온결과입니다.
추출한 데이터:
{context}

주어진 데이터의 특성을 가장 잘 표현할 수 있는 시각화 방법을 아래에서 한가지 선택하세요.
시각화 방법만 말하세요.

- 시각화 방법
1. TABLE : 범주형 데이터, 텍스트 데이터, 요약 통계 ex) 특정 시간대의 데이터 상태, 데이터 값
2. HEATMAP, BUBBLE, BAR : 빈도형 데이터, 연속형 데이터 간의 관계 ex) 온도, 압력, 출력 간의 상관관계 분석, 특정 이벤트 발생 빈도
3. SCATTER, TIMELINE, HISTOGRAM : 연속형 데이터, 시간 데이터  ex) 발전소의 일별 출력량 분포

답변: """

TemplateConfig.make_visualization_code = """
당신은 python을 이용한 데이터 시각화 전문가입니다.

plotly의 {query}을 사용하여 주어진 데이터프레임을 시각화하는 수준 높은 python code를 작성해주세요. 
무조건 python code만 생성하세요. 

다음 사항들을 반영하세요 :
1. 시각화의 유형에 관계없이 컬럼 넓이와 행 높이를 조절할 수 있도록 합니다.
2. 시각화에 제목을 추가합니다.
3. 시각화의 컬러 테마를 설정합니다.
4. 시각화의 요소 정렬 및 폰트 설정을 포함합니다.
5. 데이터프레임의 컬럼을 적절히 나누어 여러 개의 시각화로 표시합니다.
6. 시각화에 사용될 df는 가지고 있습니다.

데이터 프레임 :
{context}

출력 양식 : ```python\n<python code>\n```"""


TemplateConfig.data_analysis= """
아래 데이터는 '{query}'에 따라 데이터를 데이터베이스에서 가져온결과입니다.
추출한 데이터:
{context}

주어진 데이터의 특성을 가장 잘 표현할 수 있는 시각화 방법을 아래에서 한가지 선택하세요.
시각화 방법만 말하세요.

- 시각화 방법
1. TABLE : 범주형 데이터, 텍스트 데이터, 요약 통계 ex) 특정 시간대의 데이터 상태, 데이터 값
2. HEATMAP, BUBBLE, BAR : 빈도형 데이터, 연속형 데이터 간의 관계 ex) 온도, 압력, 출력 간의 상관관계 분석, 특정 이벤트 발생 빈도
3. SCATTER, TIMELINE, HISTOGRAM : 연속형 데이터, 시간 데이터  ex) 발전소의 일별 출력량 분포

답변: """


# 질의 분석 chain
def make_query_split_prompt(state)->ChatPromptTemplate:
    prompt = TemplateConfig.make_query_split.format_map({"query":state["query"]})
    prompt = get_template(prompt,TEMPLATE_MODEL)
    return prompt

def make_sql_query_prompt(state)->ChatPromptTemplate:
    prompt = TemplateConfig.make_sql_query.format_map({"query":state["query"],
                                                   "schema":state["schema"]})
    prompt = get_template(prompt,TEMPLATE_MODEL)
    return prompt

def make_code_prompt(state)->ChatPromptTemplate:
    prompt = TemplateConfig.make_code.format_map({"query":state["query"],
                                                  "schema":state["schema"],
                                                  "columns":state["columns"],
                                                  "sample":state["sample"]})
    prompt = get_template(prompt,TEMPLATE_MODEL)
    return prompt

def make_generate_prompt(state)->ChatPromptTemplate:
    prompt = TemplateConfig.generate.format_map({"query":state["query"],
                                                 "context":state["context"]})
    prompt = get_template(prompt,TEMPLATE_MODEL)
    return prompt

def select_visualization_prompt(state)->ChatPromptTemplate:
    prompt = TemplateConfig.select_visualization.format_map({"query":state["query"],
                                                 "context":state["context"]})
    prompt = get_template(prompt,TEMPLATE_MODEL)
    return prompt

def make_visualization_code_prompt(state)->ChatPromptTemplate:
    prompt = TemplateConfig.make_visualization_code.format_map({"query":state["query"],
                                                 "context":state["context"]})
    prompt = get_template(prompt,TEMPLATE_MODEL)
    return prompt


Chain__split_query = (RunnableLambda(make_query_split_prompt)| llm | StrOutputParser())
Chain__make_sql_query = (RunnableLambda(make_sql_query_prompt)| llm | StrOutputParser())
Chain__make_code = (RunnableLambda(make_code_prompt)| llm | StrOutputParser())
Chain__generate = (RunnableLambda(make_generate_prompt)| llm | StrOutputParser())
Chain__select_visualization = (RunnableLambda(select_visualization_prompt)| llm | StrOutputParser())
Chain__make_visualization_code = (RunnableLambda(make_visualization_code_prompt)| llm | StrOutputParser())





# 얘는 없고
# Chain__make_image = (RunnableLambda(make_image_prompt)| llm | StrOutputParser())








Chain__make_sql_query = (RunnableLambda(make_sql_query_prompt)| llm | StrOutputParser())