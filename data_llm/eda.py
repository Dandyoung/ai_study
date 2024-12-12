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


def fnc_graph(df, user_eda_column):    
    with st.container():
        st.scatter_chart(df, y=[user_eda_column])
        st.info('산점도', icon="✨")
    with st.container():
        st.plotly_chart(figure_or_data=ff.create_distplot([df[user_eda_column].dropna()], group_labels=[user_eda_column]))
        st.info('히스토그램', icon="✨")
    return

def time_series_plot(df, column_name):
    st.line_chart(df[column_name])
    st.info('시계열 플롯', icon="📈")

def box_plot(df, column_name):
    fig, ax = plt.subplots()
    sns.boxplot(y=df[column_name], ax=ax)
    st.pyplot(fig)
    st.info('박스 플롯', icon="📦")

def density_plot(df, column_name):
    fig, ax = plt.subplots()
    sns.kdeplot(df[column_name], ax=ax)
    st.pyplot(fig)
    st.info('밀도 플롯', icon="🌐")

def heatmap(df):
    corr = df.corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, ax=ax)
    st.pyplot(fig)
    st.info('히트맵', icon="🔥")


# streamlit 앱 시작
if "show_image" not in st.session_state:
    st.session_state.show_image = False
if "df" not in st.session_state:
    st.session_state.df = None
if "agentexecutor" not in st.session_state:
    st.session_state.agentexecutor = None
if "qa_result" not in st.session_state:
    st.session_state.qa_result = None
if "overlay_visible" not in st.session_state:
    st.session_state.overlay_visible = False

if "eda_df" not in st.session_state:
    st.session_state.eda_df = None

# 챗봇 대화 기록 초기화
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 버튼 클릭 이벤트 처리
if st.button("View the Workflow(Demo)"):
    st.session_state.show_image = not st.session_state.show_image

# 이미지 표시
if st.session_state.show_image:
    st.image("/workspace/youngwoo/Agent_m2m/Pages/tmp/eee.jpg", caption="View the Workflow(Demo)")

st.markdown(
    "<h1 style='text-align: center; font-size: 26px;'>👨‍👨‍👦‍👦 안녕하세요, 데이터에 관한건 <span style='color: #F23005;'>무엇이든</span> 물어보세요!</h1>",
    unsafe_allow_html=True
)
myai = MyAI(api_key=openai_api_key)

st.markdown(
    "<h1 style='text-align: center; font-size: 40px;'>🔎 Let's start with <span style='color: #D9D2D0;'>EDISON !</span></h1>",
    unsafe_allow_html=True
)

# 업로드된 파일 대신 내부 데이터프레임 사용
# st.info('수집된 가스터빈 센서 데이터입니다.', icon="💁‍♀️")
info_df = create_info_df()
for col in info_df.columns:
    if info_df[col].dtype == 'object':
        try:
            info_df[col] = pd.to_numeric(info_df[col])
        except ValueError:
            info_df[col] = info_df[col].astype(str)
# st.write(info_df.shape)
# st.dataframe(info_df)

tab1, tab2 = st.tabs(["Q & A", "EDA"])

def display_analysis_step(function, column_name, analysis_name):
    with st.spinner(f"{analysis_name} 중..."):
        result_placeholder = st.empty()
        start_message = st.markdown(f"<h6 style='color: #FFA500;'>{analysis_name} 시작</h6>", unsafe_allow_html=True)
        retval = function(column_name)
        
        full_response = ""
        for chunk in re.split(r'(\s+)', retval):
            full_response += chunk + " "
            time.sleep(0.01)
            result_placeholder.markdown(
                f"<div style='background-color:#173928; padding:10px; border-radius:5px; color:white;'> {full_response}▌</div>",
                unsafe_allow_html=True
            )
        time.sleep(0.2)
        result_placeholder.markdown(
            f"<div style='background-color:#173928; padding:10px; border-radius:5px; color:white;'> {full_response}</div>",
                unsafe_allow_html=True
        )
        start_message.markdown(f"<h6 style='color: #00FF00;'>{analysis_name} 완료 ✅</h6>", unsafe_allow_html=True)

st.markdown(
    """
    <style>
    .overlay {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(255, 255, 255, 0.8);
        z-index: 1000;
        display: none;
    }
    .spinner {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
    }
    </style>
    <div class="overlay" id="overlay">
        <div class="spinner">🔄 Loading...</div>
    </div>
    <script>
    function showOverlay() {
        document.getElementById('overlay').style.display = 'block';
    }
    function hideOverlay() {
        document.getElementById('overlay').style.display = 'none';
    }
    </script>
    """, unsafe_allow_html=True
)

def show_overlay():
    st.session_state.overlay_visible = True

def hide_overlay():
    st.session_state.overlay_visible = False

if st.session_state.overlay_visible:
    st.markdown("<script>showOverlay();</script>", unsafe_allow_html=True)
else:
    st.markdown("<script>hideOverlay();</script>", unsafe_allow_html=True)

with tab1:
    st.header("📊 Q & A")

    user_input = st.chat_input("무엇이든 물어보세요")
    if user_input:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        # Generate assistant response
        try:
            with st.spinner("잠시만 기다려 주세요, 응답을 생성 중입니다...!"):
                df = query_tool.query_dataframe(user_input, info_df)
                st.session_state.eda_df = df  # 실행된 결과를 st.session_state에 미리 저장
                agentexecutor = myai.GetAgentExecutor(df=df)
                st.session_state.agentexecutor = agentexecutor  # agentexecutor 저장
                myedahelper = MyEDAHelper(agentexecutor)
                retval = myedahelper.fnc_qa(user_input, info_df, df)
                st.session_state.qa_result = retval  # Q & A 결과를 st.session_state에 저장

                # Add assistant response to chat history
                st.session_state.chat_history.append({"role": "assistant", "content": retval})

        except Exception as e:
            error_msg = f'오류 발생: {e}'
            st.session_state.chat_history.append({"role": "assistant", "content": error_msg})

    # Display all messages
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

with tab2:
    st.header("📰 EDA")
    if st.session_state.eda_df is not None:
        selected_column = st.selectbox(
            "더 자세히 알고싶은 센서 이름을 선택해주세요.",
            ["센서 이름을 선택하세요."] + list(st.session_state.eda_df.columns),
            index=0,
            key="eda_selectbox"  # 고유한 키 설정
        )
        if selected_column and selected_column != "센서 이름을 선택하세요.":
            column_df = st.session_state.eda_df[[selected_column]].copy()  # 선택한 컬럼만 가져오기
            agentexecutor2 = myai.GetAgentExecutor(df=column_df)
            myedahelper2 = MyEDAHelper(agentexecutor2)
            result_placeholder = st.empty()
            try:
                show_overlay()
                result_placeholder.empty()  # 이전 결과를 지웁니다.

                display_analysis_step(myedahelper2.fnc_eda_summary_statistics, selected_column, "통계 분석")
                display_analysis_step(myedahelper2.fnc_eda_normality, selected_column, "정규성 검사")
                display_analysis_step(myedahelper2.fnc_eda_missing_values, selected_column, "결측치 분석")
                display_analysis_step(myedahelper2.fnc_eda_outliers, selected_column, "이상치 분석")
                # display_analysis_step(myedahelper2.fnc_eda_trends, selected_column, "추세 분석")
                
                # 추가: fnc_graph 함수 호출하여 그래프 표시
                st.subheader("📈 추가 그래프 분석")
                # fnc_graph(st.session_state.eda_df, selected_column)
                time_series_plot(st.session_state.eda_df, selected_column)
                box_plot(st.session_state.eda_df, selected_column)
                density_plot(st.session_state.eda_df, selected_column)
                heatmap(st.session_state.eda_df)
                hide_overlay()
            except Exception as e:
                st.error(f'오류 발생: {e}', icon="🚨")
    else:
        st.warning("먼저 Q & A 탭에서 질문을 통해 데이터를 불러오세요.")
