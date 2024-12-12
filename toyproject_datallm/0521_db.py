import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from langchain_openai import OpenAI
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain.agents.agent_types import AgentType
from langchain.agents.react.agent import create_react_agent
from langchain.agents import AgentExecutor, Tool
from langchain.prompts import PromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

# Prompt 파일에서 agent_prompt_en를 가져옴
from prompt_templates.prompt import agent_prompt_en

'''
에이전트 초기화
'''
def agent_init(_llm, path):

    csv_agent = create_csv_agent(llm=_llm, path=path, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION)

    # 사용할 tools 정의
    tools = [
        Tool(
            name="CSVAgent",
            func=csv_agent.invoke,
            description="'df'라는 데이터프레임을 조작하고 그래프를 그릴 수 있는 파이썬 코드를 작성하는 것이 유용합니다."
        ),
    ]
    # 사용할 답변 스키마들
    response_schemas = [
        ResponseSchema(name="output", description="사용자의 질문에 대한 자연어로 답변하세요."),
        ResponseSchema(
            name="pandas",
            description="pandas를 사용하여 질문과 관련된 파이썬 코드를 작성하세요. 적절한 코드가 생각나지 않으면, 이 부분은 비워 두세요.",
        ),
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    df = pd.read_csv(path)
    
    # max_token 떄문에 열만 넣어주는게 좋아보임..
    context_df = df.head().to_string()


    '''
    These keywords must never be translated and transformed:
    - Action:
    - Thought:
    - Action Input:
    '''
    prompt_template = PromptTemplate.from_template(
        """**INSTRUCTIONS**
        You have been given a dataset to analyze as requested by the user. 

        Please answer the user's request utilizing the tools below and answer in Korean:
        {tools}
        
        Refer to the tools as follows: {tool_names}

        Use the following format:

        Question: the input question you must answer
        Thought: your thought process of what action should be taken
        Action: stating the tool that will be used to get the desired result
        Action Input: the input into the tool
        Observation: the result of the action and what it means
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: 
        {{
            "output":"the Final Answer to the user's question",
            "pandas":"the python code for a pandas that is relevant to the question"
        }}

        **CONTEXT**
        Here is the dataframe head for context:
        {context_df}

        **QUESTION**

        Question: {input} 
        Thought: {agent_scratchpad}
        Action:"""
    )

    agent = create_react_agent(llm=_llm, tools=tools, prompt=prompt_template)
    
    # max_execution_time : 최대 응답 시간, max_iterations : 최대 반복 횟수
    agent_chain = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, handle_parsing_errors=True, max_execution_time=60, max_iterations=10, verbose=True)

    return agent_chain, output_parser, df, context_df

def execute_and_display_figure(code_string, df):
    # Create a Python script from the provided code
    temp_script_path = "/workspace/youngwoo/toyproject-datallm/tmp/temp_script.py"
    with open(temp_script_path, "w") as f:
        f.write("import pandas as pd\n")
        f.write("import json\n")
        # df를 받아서 temp_dataframe.csv 파일로 저장
        f.write("df = pd.read_csv('temp_dataframe.csv')\n")
        # f.write("df['datetime'] = pd.to_datetime(df['datetime'])\n")
        # f.write("df.set_index('datetime', inplace=True)\n")
        f.write("result = " + code_string + "\n")
        f.write("with open('/workspace/youngwoo/toyproject-datallm/tmp/temp_result.json', 'w') as result_file:\n")
        f.write("    json.dump(result.to_dict(), result_file, ensure_ascii=False, indent=4)\n")

    # df를 임시 csv파일로 저장
    df.to_csv('/workspace/youngwoo/toyproject-datallm/tmp/temp_dataframe.csv', index=False)
    # python script 실행
    os.system("python temp_script.py")


def main():
    llm = OpenAI(temperature=0.5, model="gpt-3.5-turbo-instruct", verbose=True, api_key=os.getenv("OPENAI_API_KEY"))
    agent_chain, output_parser, df, context_df = agent_init(llm, '/workspace/youngwoo/toyproject-datallm/dataset_modified.csv')

    prefix = "당신은 유능한 데이터 분석가 입니다.\n"
    question = "Question: 3월 24일 데이터에 대해 분 단위로 변환하여 기초 통계 정보를 알려주세요. (최소, 최대, 평균)"
    user_input = prefix + question

    # 에이전트 실행
    response = agent_chain.invoke({"input": user_input, "context_df": context_df})

    # 디버깅용) raw response 출력
    print("Raw Response:", response)
    try:
        parsed_response = output_parser.parse(response['output'])
        print("Answer:", parsed_response["output"])

        # 디버깅용) json 파일로 변환
        answer = parsed_response["output"]
        answer2 = parsed_response['pandas']

        output_data = {
            "output": answer,
            "pandas": answer2
        }

        with open('/workspace/youngwoo/toyproject-datallm/tmp/output.json', 'w', encoding='utf-8') as json_file:
            json.dump(output_data, json_file, ensure_ascii=False, indent=4)

        # pandas 관련 코드가 답변에 존재하면, 판다스 실행하는 코드에 붙히러 감
        if 'pandas' in parsed_response and parsed_response['pandas']:
            execute_and_display_figure(parsed_response['pandas'], df)

    except Exception as e:
        print("구문 분석 오류:", e)
        print("오류 발생 구문:", response['output'])

if __name__ == "__main__":
    main()
