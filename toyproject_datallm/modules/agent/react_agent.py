import os
import json
import pandas as pd
import matplotlib.pyplot as plt
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
# set_debug(True)
# Chain(..., verbose=True) 이렇게 원래 쓰듯이 쓰면, 해당 runalbe 객체에 대해서만 디버깅

'''
에이전트 초기화
'''

def agent_init(_llm, path, language='en'):
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

def main():
    llm = ChatOpenAI(temperature=0.6, model="gpt-4-turbo", verbose=True, api_key=os.getenv("OPENAI_API_KEY"))
    # 언어를 'ko'로 설정하여 한국어 프롬프트 사용
    agent_chain, output_parser, df, context_df, tools, tool_names, df_type = agent_init(llm, 'modified_3월24일.csv', language='ko')

    user_input = "3월 24일에 대한 데이터의 평균을 말해줘"

    response = agent_chain.invoke({"input": user_input, 
                                    "context_df": context_df, 
                                    "tools": tools, 
                                    "tool_names": tool_names, 
                                    "df_type": df_type})

    # print("Raw Response:", response)
    try:
        parsed_response = output_parser.parse(response['output'])
        print("Answer:", parsed_response["output"])

        answer = parsed_response["output"]

        output_data = {
            "output": answer,
        }

        with open('output.json', 'w', encoding='utf-8') as json_file:
            json.dump(output_data, json_file, ensure_ascii=False, indent=4)

    except Exception as e:
        print("구문 분석 오류:", e)
        print("오류 발생 구문:", response['output'])

if __name__ == "__main__":
    main()
