import os
import json
import pandas as pd
import openai
from langsmith import traceable
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain.agents.agent_types import AgentType
from langchain.agents.react.agent import create_react_agent
from langchain.agents import AgentExecutor, Tool
from langchain.prompts import PromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from prompt_templates.prompt import Prompt

# 랭체인 디버깅 하는 라이브러리
from langchain.globals import set_debug, set_verbose

set_verbose(True)

# OpenAI API 키 설정
openai.api_key = os.getenv("OPENAI_API_KEY")

# 에이전트 초기화 함수
def agent_init(_llm, path, language='ko'):
    # ZERO_SHOT_REACT_DESCRIPTION : 행동을 취하기 전에 추론 단계를 거치는 에이전트.
    csv_agent = create_csv_agent(llm=_llm, path=path, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION)

    tools = [
        Tool(
            name="CSVAgent",
            func=csv_agent.invoke,
            description="'df'라는 데이터프레임을 조작하여 사용자의 질문에 대답할 때 사용하세요."
        ),
    ]
    tool_names = [tool.name for tool in tools]

    response_schemas = [
        ResponseSchema(name="output", description="사용자의 질문에 대한 답변을 작성하세요.")
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    df = pd.read_csv(path)
    
    context_df = df.head().to_string()
    df_type = df.dtypes.to_string()

    # Prompt 클래스를 사용하여 언어별 프롬프트 가져오기
    prompt_instance = Prompt()
    agent_prompt = prompt_instance.get_prompt(language)
    prompt_template = PromptTemplate.from_template(agent_prompt)

    # ReAct Agent 생성
    agent = create_react_agent(llm=_llm, tools=tools, prompt=prompt_template)    
    agent_chain = AgentExecutor.from_agent_and_tools(agent=agent, 
                                                     tools=tools, 
                                                     handle_parsing_errors=True, 
                                                     #max_execution_time=60, 
                                                     max_iterations=10, 
                                                     verbose=True)

    return agent_chain, output_parser, df, context_df, tools, tool_names, df_type

# LangSmith를 통해 에이전트 실행을 추적하는 함수
@traceable
def run_agent(agent_chain, user_input, context_df, tools, tool_names, df_type):
    response = agent_chain.invoke({"input": user_input, 
                                   "context_df": context_df, 
                                   "tools": tools, 
                                   "tool_names": tool_names, 
                                   "df_type": df_type})
    return response

def main():
    # ChatOpenAI 인스턴스를 생성
    openai_client = ChatOpenAI(temperature=0.6, model="gpt-4-turbo", verbose=True, api_key=os.getenv("OPENAI_API_KEY"))

    # 언어를 'ko'로 설정하여 한국어 프롬프트 사용
    agent_chain, output_parser, df, context_df, tools, tool_names, df_type = agent_init(openai_client, 'modified_3월24일.csv', language='ko')

    user_input = "3월 24일에 대한 데이터의 평균을 말해줘"

    response = run_agent(agent_chain, user_input, context_df, tools, tool_names, df_type)

    try:
        # 응답을 텍스트로 처리
        response_text = response['output']
        print("Answer:", response_text)

        # 텍스트를 그대로 파일에 저장
        output_data = {
            "output": response_text,
        }

        with open('output.json', 'w', encoding='utf-8') as json_file:
            json.dump(output_data, json_file, ensure_ascii=False, indent=4)

    except Exception as e:
        print("구문 분석 오류:", e)
        print("오류 발생 구문:", response['output'])

if __name__ == "__main__":
    main()