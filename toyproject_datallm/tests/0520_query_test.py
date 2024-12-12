
import sqlite3
from langchain_openai import ChatOpenAI
from langchain.chains import create_sql_query_chain
from langchain.sql_database import SQLDatabase

# OpenAI API 설정
llm = ChatOpenAI(model="gpt-3.5-turbo")

# SQLite 데이터베이스 파일 경로
sqlite_db_path = '/workspace/youngwoo/toyproject-datallm/test_dataset.db'

# SQLite 데이터베이스 연결
conn = sqlite3.connect(sqlite_db_path)

# SQLAlchemy 엔진 생성
from sqlalchemy import create_engine
engine = create_engine(f'sqlite:///{sqlite_db_path}')

# SQLDatabase 객체 생성
db = SQLDatabase(engine)

# SQL Query Chain 생성
chain = create_sql_query_chain(llm, db)

# 질문을 통한 SQL Query Chain 실행
question = "해당 데이터셋에 몇개의 열이 있어?"
response = chain.invoke({"question": question})

print(response)
conn.close()
