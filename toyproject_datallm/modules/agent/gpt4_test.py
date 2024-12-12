import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain.agents.agent_types import AgentType
from langchain.agents import AgentExecutor, Tool
from langchain.prompts import PromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from prompt_templates.prompt import Prompt
from langchain.agents import create_openai_functions_agent
from langchain_experimental.tools import PythonREPLTool
# 랭체인 디버깅 하는 라이브러리
from langchain.globals import set_debug, set_verbose
set_verbose(True)

# def create_csv_agent_from_df(llm, df, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION):
#     csv_path = 'temp_sample.csv'
#     df.to_csv(csv_path, index=False)
#     return create_csv_agent(llm=llm, path=csv_path, agent_type=agent_type)

def agent_init(_llm, sample_df, language='en'):
    # CSV 에이전트 생성
    # csv_agent = create_csv_agent_from_df(llm=_llm, df=sample_df, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION)

    tools = [
        PythonREPLTool(description="Use this to write Python code for the user's question.")
    ]
    tool_names = [tool.name for tool in tools]

    response_schemas = [
        ResponseSchema(name="output", description="Find the answer to respond in natural language."),
        ResponseSchema(
            name="pandas",
            description="Write Python code related to the question using pandas. If you can't think of appropriate code, leave this part blank.",
        ),
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    
    # context_df의 크기를 줄입니다.
    context_df = sample_df.to_string()

    # Prompt 클래스를 사용하여 언어별 프롬프트 가져오기
    prompt_instance = Prompt()
    agent_prompt = prompt_instance.get_prompt(language)
    
    # 시스템 프롬프트 추가
    system_prompt = "You are a skilled data analyst.\n"
    combined_prompt = system_prompt + agent_prompt
    prompt_template = PromptTemplate.from_template(combined_prompt)

    # OpenAI Functions Agent 생성
    agent = create_openai_functions_agent(llm=_llm, tools=tools, prompt=prompt_template)
    
    agent_chain = AgentExecutor(agent=agent, tools=tools, handle_parsing_errors=True, max_execution_time=60, max_iterations=10, verbose=True)

    return agent_chain, output_parser, context_df, tools, tool_names

def execute_and_display_figure(code_string, df):
    temp_script_path = os.path.join(os.getcwd(), "temp_script.py")
    temp_csv_path = os.path.join(os.getcwd(), "temp_dataframe.csv")
    temp_result_path = os.path.join(os.getcwd(), "temp_result.json")

    with open(temp_script_path, "w") as f:
        f.write("import pandas as pd\n")
        f.write("import json\n")
        f.write(f"df = pd.read_csv('{temp_csv_path}')\n")
        f.write("result = " + code_string + "\n")
        f.write("if isinstance(result, pd.DataFrame):\n")
        f.write("    result = result.to_dict()\n")
        f.write(f"with open('{temp_result_path}', 'w') as result_file:\n")
        f.write("    json.dump(result, result_file, ensure_ascii=False, indent=4)\n")

    df.to_csv(temp_csv_path, index=False)
    os.system(f"python {temp_script_path}")

def main():
    llm = ChatOpenAI(temperature=0.3, model="gpt-4", verbose=True, api_key=os.getenv("OPENAI_API_KEY"), max_tokens=1000)

    # CSV 파일에서 일부 데이터만 읽어옵니다.
    df = pd.read_csv('temp_dataframe.csv')
    sample_df = df.head(2)  # 필요한 데이터만 샘플링

    # 언어를 'ko'로 설정하여 한국어 프롬프트 사용
    agent_chain, output_parser, context_df, tools, tool_names = agent_init(llm, sample_df, language='gpt_en')

    # prefix = "You are a skilled data analyst.\n"
    question = "Question: Tell me the average of the data for March 24th."
    # user_input = prefix + question

    response = agent_chain.invoke({
        "input": question,
        "context_df": context_df,
        "tools": tools,
        "tool_names": tool_names
    })

    # response 출력 확인용

    print(response)

    # try:
    #     parsed_response = output_parser.parse(response['output'])
    #     print("Answer:", parsed_response["output"])

    #     answer = parsed_response["output"]
    #     answer2 = parsed_response['pandas']

    #     output_data = {
    #         "question":question,
    #         "output": answer,
    #         "pandas": answer2
    #     }

    #     output_path = os.path.join(os.getcwd(), 'output.json')
    #     with open(output_path, 'w', encoding='utf-8') as json_file:
    #         json.dump(output_data, json_file, ensure_ascii=False, indent=4)

    #     # if 'pandas' in parsed_response and parsed_response['pandas']:
    #     #     execute_and_display_figure(parsed_response['pandas'], df)

    # except Exception as e:
    #     print("구문 분석 오류:", e)
    #     print("오류 발생 구문:", response['output'])

if __name__ == "__main__":
    main()
