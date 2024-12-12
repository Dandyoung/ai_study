'''
실행방법 : streamlit run eda.py 
'''

import streamlit as st
import time
import re
import pandas as pd
import numpy as np
import plotly.figure_factory as ff
import seaborn as sns
import matplotlib.pyplot as plt
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentExecutor
from MyAI import MyAI
from MyEDAHelper import MyEDAHelper
import sys
import os

# 모듈 검색 경로에 /workspace/youngwoo/Agent_m2m/Pages 추가
sys.path.append(os.path.abspath("/workspace/youngwoo/Agent_m2m/Pages"))
# default_dataframe에서 create_info_df 함수 가져오기
from default_dataframe import create_info_df

from query_dataframe import QueryDataFrame
db_path = '/workspace/youngwoo/Agent_m2m/Pages/dataset/most_recent.db'
query_tool = QueryDataFrame(db_path, openai_api_key)


def fnc_graph(user_eda_column):    
    with st.container():
        st.scatter_chart(df, y=[user_eda_column])
        st.info('산점도', icon="✨")
    with st.container():
        st.plotly_chart(figure_or_data=ff.create_distplot([df[user_eda_column].dropna()], group_labels=[user_eda_column]))
        st.info('히스토그램', icon="✨")
    return

'''
여기부터 모듈화 해야함 테스트용
'''
from ml_collections import ConfigDict
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import pandas as pd
from langchain_core.runnables import RunnableLambda
from loguru import logger
import subprocess
import re
from ml_collections import ConfigDict
from langchain_openai import ChatOpenAI
import json

def get_input_gpt(user_message):
    return user_message

def get_template(template,var="custom"):
    if var == "gpt":
        return get_input_gpt(template)

TemplateConfig = ConfigDict()

TemplateConfig.visualization_prompt  = '''
당신은 python을 이용한 데이터 시각화 전문가입니다.

아래 데이터는 '{query}'에 따라 데이터를 데이터베이스에서 가져온 결과입니다.
추출한 데이터:
{context}

먼저, 주어진 데이터의 특성을 가장 잘 표현할 수 있는 시각화 방법을 아래에서 한 가지 선택하세요.

- 시각화 방법
TABLE : 범주형 데이터, 텍스트 데이터, 요약 통계 ex) 특정 시간대의 데이터 상태, 데이터 값
HEATMAP, BUBBLE, BAR : 빈도형 데이터, 연속형 데이터 간의 관계 ex) 온도, 압력, 출력 간의 상관관계 분석, 특정 이벤트 발생 빈도
SCATTER, TIMELINE, HISTOGRAM : 연속형 데이터, 시간 데이터 ex) 발전소의 일별 출력량 분포

선택한 시각화 방법을 기반으로 plotly를 사용하여 주어진 데이터프레임을 시각화하는 수준 높은 python code를 작성해주세요.
무조건 python code만 생성하세요.

다음 사항들을 반영하세요 :

1. 시각화의 유형에 관계없이 컬럼 넓이와 행 높이를 조절할 수 있도록 합니다.
2. 시각화에 제목을 추가합니다.
3. 시각화의 컬러 테마를 설정합니다.
4. 시각화의 요소 정렬 및 폰트 설정을 포함합니다.
5. 데이터프레임의 컬럼을 적절히 나누어 여러 개의 시각화로 표시합니다.
6. df는 정의되어 있습니다.
7. 생성된 시각화 이미지는 './tmp_img' 폴더에 저장되게 해주세요.

출력 양식 :
```python\n<python code>\n```
'''

llm = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo')
TEMPLATE_MODEL = 'gpt'

def make_visualization_prompt(state)->ChatPromptTemplate:
    prompt = TemplateConfig.visualization_prompt.format_map({"query":state["query"],
                                                                "context":state["context"]})
    prompt = get_template(prompt,TEMPLATE_MODEL)
    return prompt


# with st.sidebar:
#     st.title("💀 Hi, I am :blue[Agent_m2m] ",)

#     openai_api_key = st.text_input(
#         "OpenAI API Key", key="langchain_search_api_key_openai", type="password"
#     )
#     if openai_api_key:
#         myai:MyAI = MyAI(api_key=openai_api_key)
#         if myai:
#             try:
#                 myai.ValidateLLM()    
#                 st.write('You are using LLM: '+ myai.GetLLM().model_name)                
#             except Exception as e:
#                 #st.write(e)
#                 st.error('Please enter a valid  OpenAI API key to continue.', icon="🚨")
#         else:
#             #st.write('Please add valid  OpenAI API key to continue.')=
#             st.error('Please enter a valid  OpenAI API key to continue.', icon="🚨")
        

#     "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
#     "[View the source code](https://github.com/cdebdeep/Agent_m2m.git)"
#     "[Open in GitHub Codespaces](https://ideal-guacamole-q47g9p6xrj24xgj.github.dev/)"

# st.title("🔎 EDA - With your file")


# streamlit 앱 시작
st.markdown(
    "<h1 style='text-align: center; font-size: 35px;'>👨‍👨‍👦‍👦 안녕하세요, 저는 <span style='color: #A61717;'>스마트 엠투엠</span> 입니다</h1>", 
    unsafe_allow_html=True
)
# OpenAI API 키가 미리 선언되어 있으므로 바로 진행
myai = MyAI(api_key=openai_api_key)

st.markdown(
    "<h1 style='text-align: center; font-size: 28px;'>🔎 Let's start with <span style='color: #D9D2D0;'>EDA (Exploratory Data Analysis)!</span></h1>", 
    unsafe_allow_html=True
)


# 업로드된 파일 대신 내부 데이터프레임 사용
st.info('수집된 가스터빈 센서 데이터입니다.', icon="💁‍♀️")
info_df = create_info_df()                           
# 데이터프레임의 각 열 타입을 확인하고 변환
for col in info_df.columns:
    if info_df[col].dtype == 'object':
        try:
            info_df[col] = pd.to_numeric(info_df[col])
        except ValueError:
            info_df[col] = info_df[col].astype(str)

st.write(info_df.shape)
st.dataframe(info_df)


tab1, tab2, tab3, tab4 = st.tabs(["Q & A", "EDA","EDA-그래프", "생성"])

with tab1:
    st.header("📊 Q & A  ")
    question1 = st.text_input(
        "무엇이든 물어보세요",
        placeholder="예: 24일 가스관련 데이터의 평균값은 얼마입니까?",
        disabled=False,
    )
    if question1:
        try:
            with st.spinner("잠시만 기다려 주세요, 응답을 생성 중입니다... !"):
                # 여기서 받는 df가 쿼리 실행된 데이터
                df = query_tool.query_dataframe(question1, info_df)
                agentexecutor = myai.GetAgentExecutor(df=df)
                myedahelper = MyEDAHelper(agentexecutor)
                retval = myedahelper.fnc_qa(question1,info_df, df)

                logger.debug("체인 실행 중..")
                Chain__split_query = (RunnableLambda(make_visualization_prompt)| llm | StrOutputParser())
                tmp = Chain__split_query.invoke({"query": question1,"context": df}).strip()
                start_idx = tmp.find("```python") + len("```python")
                end_idx = tmp.find("```", start_idx)
                tmp_python_code = tmp[start_idx:end_idx].strip()

                print(tmp_python_code)
                # Execute the Python code
                # local_vars = {}
                # exec(tmp_python_code, {"df": df}, local_vars)

                temp_dir = "/workspace/youngwoo/Agent_m2m/Pages/tmp"
                temp_script_path = os.path.join(temp_dir, "temp_script.py")

                with open(temp_script_path, "w") as f:
                    f.write(tmp_python_code)

                # print(f"retval :{retval}")
                st.success(retval, icon="✅")
        except Exception as e:
            st.error('표시할 데이터가 없습니다 !!.', icon="🚨")    

# with tab2:
#     st.header("EDA")
#     question2 = st.text_input(
#         "EDA를 위해 데이터 열 이름을 입력하세요",
#         placeholder="열 이름을 입력하세요?",
#         disabled=False,
#     )
#     if question2:
#         try:
#             with st.spinner("잠시만 기다려 주세요, 응답을 생성 중입니다... !"):                  
#                 st.success('다음은 응답입니다:', icon="✅")
#                 myedahelper = MyEDAHelper(agentexecutor)
#                 retval = myedahelper.fnc_eda(question2)
#                 message_placeholder = st.empty()                

#                 # Simulate stream of response with milliseconds delay
#                 full_response = ""
#                 for chunk in re.split(r'(\s+)', retval):
#                     full_response += chunk + " "
#                     time.sleep(0.01)

#                     # Add a blinking cursor to simulate typing
#                     message_placeholder.markdown(full_response + "▌")

#         except Exception as e:
#             st.error('표시할 데이터가 없습니다 !!.', icon="🚨")

# with tab3:
#     st.header("EDA-그래프")
#     question3 = st.text_input(
#         "EDA 그래프를 위해 데이터 열 이름을 입력하세요",
#         placeholder="열 이름을 입력하세요?",
#         disabled=False,
#     )
#     if question3:
#         try:
#             with st.spinner("잠시만 기다려 주세요, 응답을 생성 중입니다... !"):
#                 fnc_graph(question3)
#         except Exception as e:
#             st.error('표시할 데이터가 없습니다 !!.', icon="🚨")            

# with tab4:
#     st.header("생성")
#     question4 = st.text_input(
#         "여기에 쿼리를 입력하세요",
#         placeholder="예: '열 이름이 10보다 작은 경우?",
#         disabled=False,
#     )
#     if question4:
#         try:
#             with st.spinner("잠시만 기다려 주세요, 응답을 생성 중입니다... !"):
#                 myedahelper = MyEDAHelper(agentexecutor)
#                 retval = myedahelper.fnc_modifydata(question4)
#                 if not retval.empty:
#                     st.success('전체 행과 열의 수는 다음과 같습니다:', icon="✅")
#                     st.write(retval.shape)
#                     st.write(retval)
#         except Exception as e:
#             st.error(e, icon="🚨")

# st.write("방문해 주셔서 감사합니다!")

