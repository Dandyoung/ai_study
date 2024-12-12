import pandas as pd
import numpy as np
import plotly.figure_factory as ff
import scipy as sp
import re

from langchain_openai import ChatOpenAI
from langchain.agents.agent_types import AgentType
from langchain.agents import AgentExecutor

from langchain_experimental.agents import create_pandas_dataframe_agent



class MyEDAHelper:
    def __init__(self, agent):
        self.agent = agent
    
    def generate_prompt(self, user_question, info_df, df):
        qa_prompt = f'''
        당신은 뛰어난 데이터 분석가입니다. 
        제공된 데이터 프레임은 2023년 '가스터빈 1호기' 관련 데이터들로, 
        아래의 데이터 기본 정보와 사용자 질문과 질문과 관련된 각 datetime 별 데이터를 통해 '{user_question}'에 대해 한국어로 답변하세요.

        - 데이터 프레임 기본 정보 :
        {info_df}
        - 질문과 관련된 각 datetime 별 데이터 : 
        {df}
        '''
        return qa_prompt

    def fnc_qa(self, user_question, info_df, df):
        # 프롬프트 생성
        prompt = self.generate_prompt(user_question, info_df, df)
        
        # 디버깅을 위해 생성된 프롬프트 출력
        print("Generated Prompt:\n", prompt)
        
        # 생성된 프롬프트를 사용하여 agent 실행
        result = self.agent.run(prompt)    
        return result


    def fnc_eda(self, user_eda_column):
        # 주어진 컬럼의 평균, 중앙값, 최빈값, 표준편차, 분산, 범위, 사분위수, 왜도, 첨도를 계산합니다.
        summary_statistics = self.agent.run(f"{user_eda_column} 컬럼의 평균, 중앙값, 최빈값, 표준편차, 분산, 범위, 사분위수, 왜도, 첨도는 무엇인가요? 반드시 한국어로 답변해주세요.")
        
        # 주어진 컬럼의 정규성 또는 특정 분포 형태를 검사합니다.
        normality = self.agent.run(f"{user_eda_column} 컬럼의 정규성 또는 특정 분포 형태를 검사하세요. 반드시 한국어로 답변해주세요.")
        
        # 주어진 컬럼의 이상치 존재 여부를 평가합니다.
        outliers = self.agent.run(f"{user_eda_column} 컬럼의 이상치 존재 여부를 평가하세요. 반드시 한국어로 답변해주세요.")
        
        # 주어진 컬럼의 추세, 계절성, 주기 패턴을 분석합니다.
        trends = self.agent.run(f"{user_eda_column} 컬럼의 추세, 계절성, 주기 패턴을 분석하세요. 반드시 한국어로 답변해주세요.")
        
        # 주어진 컬럼의 결측값 정도를 확인합니다.
        missing_values = self.agent.run(f"{user_eda_column} 컬럼의 결측값 정도를 확인하세요. 반드시 한국어로 답변해주세요.")
        
        return summary_statistics + "\n\n " + normality + "\n\n " + outliers + "\n\n " + trends + "\n\n " + missing_values

    # 주어진 컬럼의 결측값 정도를 확인합니다.
    def fnc_eda_missing_values(self, user_eda_column):
        missing_values = self.agent.run(f"{user_eda_column} 컬럼의 결측값 정도를 확인하세요. 반드시 한국어로 답변해주세요.")
        return missing_values

    # 주어진 컬럼의 평균, 중앙값, 최빈값, 표준편차, 분산, 범위, 사분위수, 왜도, 첨도를 계산합니다.
    def fnc_eda_summary_statistics(self, user_eda_column):
        summary_statistics = self.agent.run(f"{user_eda_column} 컬럼의 평균, 중앙값, 최빈값, 표준편차, 분산, 범위, 사분위수, 왜도, 첨도는 무엇인가요? 반드시 한국어로 답변해주세요.")
        return summary_statistics

    # 주어진 컬럼의 정규성 또는 특정 분포 형태를 검사합니다.
    def fnc_eda_normality(self, user_eda_column):
        normality = self.agent.run(f"{user_eda_column} 컬럼의 정규성 또는 특정 분포 형태를 검사하세요. 반드시 한국어로 답변해주세요.")
        return normality

    # 주어진 컬럼의 이상치 존재 여부를 평가합니다.
    def fnc_eda_outliers(self, user_eda_column):
        outliers = self.agent.run(f"{user_eda_column} 컬럼의 이상치 존재 여부를 평가하세요. 반드시 한국어로 답변해주세요.")
        return outliers

    # 주어진 컬럼의 추세, 계절성, 주기 패턴을 분석합니다.
    def fnc_eda_trends(self, user_eda_column):
        trends = self.agent.run(f"{user_eda_column} 컬럼의 추세, 계절성, 주기 패턴을 분석하세요. 반드시 한국어로 답변해주세요.")
        return trends

    
    def fnc_modifydataAsRaw(self,modify_query)->str:

        agent_executor:AgentExecutor=self.agent
        
        return agent_executor.run(modify_query)
    
    
    def fnc_modifydata(self,modify_query)->pd.DataFrame:

        agent_executor:AgentExecutor=self.agent
        str = agent_executor.run(modify_query)

        # Extract rows of data
        rowswithheader = re.findall(r'\|.*\|.*\|', str)
        #print(rowswithheader)
        rows = re.findall(r'\| *\d+ *\|.*\|', str)
        #print(rows)
        # Extract column headers from the first row
        columns = re.findall(r' *([^|]+) *\|', rowswithheader[0])
        #print(columns)
        # Prepare data for DataFrame
        data = []
        for row in rows[0:]:
            values = re.findall(r' *([^|]+) *\|', row)
            data.append(values)
        #print(data)
        # Create DataFrame
        df = pd.DataFrame(data, columns=columns)
        
        return df
    

    
