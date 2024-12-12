import os
import sys
from langchain_openai import ChatOpenAI
from langchain.chains.sql_database.query import create_sql_query_chain
from langchain_community.utilities import SQLDatabase
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import sqlite3

# Helper functions
def get_schema(_):
    return db.get_table_info()

def run_query(query):
    return db.run(query)

# Database path
db_path = "/workspace/youngwoo/toyproject-datallm/modules/agent/tmp_dataset/test_database.db"

# Initialize the database connection
db = SQLDatabase.from_uri(f"sqlite:///{db_path}")

# Initialize the language model
model = ChatOpenAI(temperature=0, model_name='gpt-4', api_key=os.getenv("OPENAI_API_KEY"))

# Prompt for generating a SQL query
template_query = """
테이블 스키마를 바탕으로 사용자의 질문에 답하는 SQLite 쿼리를 작성하세요:
table 이름은 test_table 입니다.
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

def main():
    question = "3월 24일 데이터에 대해 분 단위로 변환하여 기초 통계 정보를 알려주세요. (최소, 최대, 평균)"

    # Generate SQL query from natural language question
    sql_query = generate_sql_query(question)
    print("Generated SQL Query:", sql_query)

    # Execute the SQL query
    columns, results = execute_sql_query(db_path, sql_query)
    print("SQL Query Result Columns:", columns)
    for row in results:
        print(row)

if __name__ == "__main__":
    main()
