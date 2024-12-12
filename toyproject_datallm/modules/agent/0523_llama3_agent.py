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
from prompt_templates.prompt import Prompt
from langchain_experimental.tools import PythonREPLTool
from langchain_community.chat_models import ChatAnthropic
from langchain.agents.react.agent import create_react_agent

# 랭체인 디버깅 하는 라이브러리
from langchain.globals import set_debug, set_verbose
set_verbose(True)

def agent_init(_llm, path, language='en'):
    # CSV 에이전트 생성
    csv_agent = create_csv_agent(llm=_llm, path=path, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION)

    # 도구 목록에 PythonREPLTool 추가
    tools = [
        Tool(
            name="CSVAgent",
            func=csv_agent.invoke,
            description="'df'라는 데이터프레임을 조작하여 사용자의 질문을 파이썬 코드로 작성할 때 사용하세요."
        ),
        
        PythonREPLTool(description="사용자의 질문에 대해 파이썬 코드를 작성해야할 때 사용하세요.") 
    ]

    response_schemas = [
        ResponseSchema(name="output", description="사용자의 질문에 대한 답변을 자연어로 말하세요."),
        ResponseSchema(
            name="pandas",
            description="pandas를 사용하여 질문과 관련된 파이썬 코드를 작성하세요. 적절한 코드가 생각나지 않으면, 이 부분은 비워 두세요.",
        ),
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    df = pd.read_csv(path)
    
    context_df = df.head().to_string()

    # Prompt 클래스를 사용하여 언어별 프롬프트 가져오기
    prompt_instance = Prompt()
    agent_prompt = prompt_instance.get_prompt(language)
    prompt_template = PromptTemplate.from_template(agent_prompt)

    # ReAct Agent 생성
    agent = create_react_agent(llm=_llm, tools=tools, prompt=prompt_template)
    
    agent_chain = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, handle_parsing_errors=True, max_execution_time=60, max_iterations=10, verbose=True)

    return agent_chain, output_parser, df, context_df

def execute_and_display_figure(code_string, df):
    temp_script_path = os.path.join(os.getcwd(), "temp_script.py")
    temp_csv_path = os.path.join(os.getcwd(), "temp_dataframe.csv")
    temp_result_path = os.path.join(os.getcwd(), "temp_result.json")

    with open(temp_script_path, "w") as f:
        f.write("import pandas as pd\n")
        f.write("import json\n")
        f.write(f"df = pd.read_csv('{temp_csv_path}')\n")
        f.write("result = " + code_string + "\n")
        f.write(f"with open('{temp_result_path}', 'w') as result_file:\n")
        f.write("    json.dump(result.to_dict(), result_file, ensure_ascii=False, indent=4)\n")

    df.to_csv(temp_csv_path, index=False)
    os.system(f"python {temp_script_path}")

def main():
    llm = OpenAI(temperature=0.5, model="gpt-3.5-turbo-instruct", verbose=True, api_key=os.getenv("OPENAI_API_KEY"))
    agent_chain, output_parser, df, context_df = agent_init(llm, 'temp_dataframe.csv', language='ko')

    prefix = "당신은 df로 정의된 데이터 프레임을 조작하고 있습니다.\n"
    question = "Question: 3월 24일의 각 열의 데이터에 대한 평균"
    user_input = prefix + question
    context_df += df.dtypes

    response = agent_chain.invoke({"input": user_input, "context_df": context_df})

    print("Raw Response:", response)
    
    # 도구 사용 로그 출력
    for action in response.get("actions", []):
        print(f"Tool used: {action['tool']} with input: {action['tool_input']}")

    try:
        parsed_response = output_parser.parse(response['output'])
        print("Answer:", parsed_response["output"])

        answer = parsed_response["output"]
        answer2 = parsed_response['pandas']

        output_data = {
            "output": answer,
            "pandas": answer2
        }

        with open('output.json', 'w', encoding='utf-8') as json_file:
            json.dump(output_data, json_file, ensure_ascii=False, indent=4)

        if 'pandas' in parsed_response and parsed_response['pandas']:
            execute_and_display_figure(parsed_response['pandas'], df)

    except Exception as e:
        print("구문 분석 오류:", e)
        print("오류 발생 구문:", response['output'])

if __name__ == "__main__":
    main()