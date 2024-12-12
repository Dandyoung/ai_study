import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from langchain_openai import OpenAI
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain.agents.agent_types import AgentType
from langchain.agents import AgentExecutor, Tool
from langchain.prompts import PromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.agents import create_react_agent
from prompt_templates.prompt_test import Prompt
from langchain_openai import ChatOpenAI
# 랭체인 디버깅 하는 라이브러리
from langchain.globals import set_debug, set_verbose

# 도구용 라이브러리
from langchain_experimental.utilities import PythonREPL
# from langchain.chains.sql_database.query import create_sql_query_chain
# from langchain_community.utilities import SQLDatabase

set_verbose(True)
# set_debug(True)
# Chain(..., verbose=True) 이렇게 원래 쓰듯이 쓰면, 해당 runalbe 객체에 대해서만 디버깅

'''
에이전트 초기화
'''

def agent_init(_llm, path, language):
    # ZERO_SHOT_REACT_DESCRIPTION : 행동을 취하기 전에 추론 단계를 거치는 에이전트.
    #description="사용자의 질문에 대한 Python 코드를 작성하기 위해 'df' 데이터프레임을 사용하세요."

    csv_agent = create_csv_agent(llm=_llm, path=path, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION)
    python_repl = PythonREPL()

    tools = [
        Tool(
            name="CSVAgent",
            func=csv_agent.invoke,
            description="Use the 'df' dataframe to write Python code for the user's question."
        ),
        Tool(
            name="python_repl",
            description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.",
            func=python_repl.run,
        )
    ]

    tool_names = [tool.name for tool in tools]

    response_schemas = [
        ResponseSchema(name="output", 
                       description="Analyze the df data related to the user's question and respond in natural language."
                       # description="사용자의 질문과 관련된 'df' 데이터를 분석하고 자연어로 응답하세요."
                       ),
        ResponseSchema(
            name="pandas",
            description="Write Python code related to the question using pandas."
            # description="pandas를 사용하여 질문과 관련된 Python 코드를 작성하세요."
        )
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    df = pd.read_csv(path)

    context_df = df.head().to_string()
    tpye_df = df.dtypes

    # Prompt 클래스를 사용하여 언어별 프롬프트 가져오기
    prompt_instance = Prompt()
    agent_prompt = prompt_instance.get_prompt(language)
    
    # 시스템 프롬프트 추가
    #system_prompt = "당신은 유능한 데이터 분석가입니다.\n"


    system_prompt = "You are a skilled data analyst.\n Please ensure that the query and answer exist, and improve it into a graph that can alleviate the answer.\n"
    combined_prompt = system_prompt + agent_prompt
    prompt_template = PromptTemplate.from_template(combined_prompt)

    # ReAct Agent 생성
    agent = create_react_agent(llm=_llm, tools=tools, prompt=prompt_template)
    agent_chain = AgentExecutor.from_agent_and_tools(agent=agent, 
                                                     tools=tools, 
                                                     handle_parsing_errors=True, 
                                                     max_execution_time=60, 
                                                     max_iterations=10, 
                                                     verbose=True)

    return agent_chain, output_parser, df, context_df, tools, tool_names , tpye_df

def execute_and_display_figure(code_string, df):
    temp_dir = "/workspace/youngwoo/toyproject-datallm/modules/agent/tmp"
    temp_script_path = os.path.join(temp_dir, "temp_script.py")
    temp_csv_path = os.path.join(temp_dir, "temp_dataframe.csv")

    with open(temp_script_path, "w") as f:
        f.write("import pandas as pd\n")
        f.write(f"df = pd.read_csv('{temp_csv_path}')\n")
        f.write(code_string)

    df.to_csv(temp_csv_path, index=False)
    # os.system(f"python {temp_script_path}")
    # os.system(f"python {temp_script_path}")

def main():
    llm = ChatOpenAI(temperature=0.1,
                    model="gpt-4-turbo", 
                    verbose=True, 
                    api_key=os.getenv("OPENAI_API_KEY"))
    agent_chain, output_parser, df, context_df, tools, tool_names, df_type = agent_init(llm, 'gt_v1_head.csv', language='ko')

    #prefix = "당신은 유능한 데이터 분석가입니다.\n"
    #question = "3월 24일 데이터의 최대값, 최소값, 평균값을 말해줘."

    prefix = "You are a skilled data analyst.\n"
    # question = "Tell me the maximum, minimum, average of the data for March 24th."
    question = "Please convert the data from March 24th to minute intervals and provide basic statistical information"
    user_input = prefix + question

    response = agent_chain.invoke({
        "input": user_input,
        "context_df": context_df,
        "tools": tools,
        "tool_names": tool_names,
        "df_type" : df_type
    })

    # print("Raw Response:", response)
    try:
        parsed_response = output_parser.parse(response['output'])
    
        answer = parsed_response["output"]
        output_data = {
            "output": answer
        }

        # print("Answer:", parsed_response["output"])
        with open('/workspace/youngwoo/toyproject-datallm/modules/agent/tmp/output.json', 'w', encoding='utf-8') as json_file:
            json.dump(output_data, json_file, ensure_ascii=False, indent=4)

        if 'pandas' in parsed_response and parsed_response['pandas']:
            print("finish")
            # execute_and_display_figure(parsed_response['pandas'], df)

    except Exception as e:
        print("exception :", e)
        print("오류 발생 이유:", response['output'])

if __name__ == "__main__":
    main()
