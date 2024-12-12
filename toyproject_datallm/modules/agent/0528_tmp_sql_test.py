import os
import sys
from langchain_openai import ChatOpenAI
from langchain.chains.sql_database.query import create_sql_query_chain
from langchain_community.utilities import SQLDatabase
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import sqlite3
import tiktoken  # OpenAI의 tiktoken 라이브러리 사용

# Helper functions
def get_schema(_):
    return db.get_table_info()

def run_query(query):
    return db.run(query)

# Initialize the database connection
db = SQLDatabase.from_uri("sqlite:////workspace/youngwoo/toyproject-datallm/modules/agent/tmp_dataset/test_database.db")

# Initialize the language model
model = ChatOpenAI(temperature=0, model_name='gpt-4', api_key=os.getenv("OPENAI_API_KEY"))

# Prompt for generating a SQL query
template_query = """
테이블 스키마를 바탕으로 사용자의 질문에 답하는 PostgreSQL 쿼리를 작성하세요:
{schema}

Question: {question}
SQL Query:"""

prompt_query = ChatPromptTemplate.from_template(template_query)

# SQL query generation chain
sql_response = (
    RunnablePassthrough.assign(schema=get_schema) 
    | prompt_query
    | model.bind(stop=["\nSQLResult:"])
    | StrOutputParser()
)

# Prompts for generating the final answer by running a SQL query on DB
template_response = """
아래의 schema, data, Question을 바탕으로 주어진 최소, 최대, 평균, 중앙값, Q1, Q3, IQR을 계산하여 튜플 값으로 저장된 데이터를 분석하여 자연어 응답을 작성하세요:
{schema}

**data** 
1. datetime을 인덱스로 설정하여 5분 간격으로 데이터를 묶었습니다. (시작시간 저장)
2. 각 숫자형 데이터 컬럼에 대해 최소, 최대, 평균, 중앙값, Q1, Q3, IQR을 계산하여 튜플 값으로 저장되어 있습니다.
3. object 타입 컬럼에 대해 5분 사이 값이 변경되면 1, 아니면 0으로 설정되어 있습니다.

data : {scratch_pad}
Question: {question}
Response: {response}"""

second_template = """
아래의 schema, previous Response를 바탕으로 주어진 data를 분석하여 자연어 응답을 생성하세요.
{schema}

**data** 
1. datetime을 인덱스로 설정하여 5분 간격으로 데이터를 묶어져 있습니다. (시작시간 저장)
2. 각 숫자형 데이터 컬럼에 대해 최소, 최대, 평균, 중앙값, Q1, Q3, IQR을 계산하여 튜플 값으로 저장되어 있습니다.
3. object 타입 컬럼에 대해 5분 사이 값이 변경되면 1, 아니면 0으로 설정되어 있습니다.
4. 각 데이터는 토큰 제한에 따라 잘렸을 수 있으니, 주의하세요.

**previous Response**
- 이전 data에 대한 분석입니다. 답변 생성 시 참고하세요.

data : {scratch_pad}
Question: {question}
previous Response: {previous_response}
Response: {response}
"""

prompt_response = ChatPromptTemplate.from_template(template_response)
second_prompt_response = ChatPromptTemplate.from_template(second_template)

# Function to generate SQL query from natural language question
def generate_sql_query(question):
    response = sql_response.invoke({"question": question})
    return response

# Function to execute SQL query and get the result
def execute_sql_query(db_path, query):
    # 데이터베이스에 연결
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 쿼리 실행
    cursor.execute(query)
    
    # 결과 가져오기
    results = cursor.fetchall()
    
    # 열 이름 가져오기
    columns = [description[0] for description in cursor.description]

    # 연결 닫기
    conn.close()
    
    return columns, results

# Tokenize text data
def tokenize_text(text, encoding_name="cl100k_base"):
    encoding = tiktoken.get_encoding(encoding_name)
    tokens = encoding.encode(text)
    return tokens

# Decode token data
def decode_tokens(tokens, encoding_name="cl100k_base"):
    encoding = tiktoken.get_encoding(encoding_name)
    text = encoding.decode(tokens)
    return text

# Function to generate natural language response
def generate_natural_language_response(schema, question, query, response, previous_response=None):
    all_data = response
    all_data_text = "\n".join(str(row) for row in all_data)
    all_tokens = tokenize_text(all_data_text)

    chunk_size = 1000  # 토큰 크기 설정
    num_chunks = (len(all_tokens) + chunk_size - 1) // chunk_size

    final_response = ""

    for i in range(num_chunks):
        scratch_pad_tokens = all_tokens[i*chunk_size:(i+1)*chunk_size]
        scratch_pad = decode_tokens(scratch_pad_tokens)
        
        # 디버깅을 위해 scratch_pad 출력
        # print(f"Chunk {i+1}/{num_chunks}:")
        # print(scratch_pad[:1000])  # 토큰 수가 많을 수 있으므로 처음 1000자만 출력
        
        prompt_input = {
            "schema": schema,
            "scratch_pad": scratch_pad,
            "question": question,
            "response": previous_response if previous_response else ""
        }

        if i == 0:
            formatted_prompt = prompt_response.format(**prompt_input)
        else:
            prompt_input["previous_response"] = previous_response  # 이전 결과를 previous_response로 추가
            prompt_input["response"] = ""
            formatted_prompt = second_prompt_response.format(**prompt_input)
        
        current_response = model.invoke(formatted_prompt)
        print(f"Chunk {i+1}/{num_chunks} 이전 답변 : {previous_response}")
        print({'\n' * 20})
        print(f"Chunk {i+1}/{num_chunks} 현재 답변 : {current_response}")
        
        previous_response = current_response

    return previous_response

def main():
    question = "3월 24일의 데이터 중 12시부터 1시까지의 데이터를 분석해줘"

    # Generate SQL query from natural language question
    sql_query = generate_sql_query(question)
    print("Generated SQL Query:", sql_query)

    # Execute the SQL query
    db_path = "/workspace/youngwoo/toyproject-datallm/modules/agent/tmp_dataset/test_database.db"
    columns, results = execute_sql_query(db_path, sql_query)
    # print("SQL Query Result Columns:", columns)
    # for row in results:
    #     print(row)

    # Generate final natural language response
    schema = get_schema(None)
    final_response = generate_natural_language_response(schema, question, sql_query, results)
    print("Final Response:", final_response)

if __name__ == "__main__":
    main()
