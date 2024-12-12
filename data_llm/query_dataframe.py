import pandas as pd
import sqlite3
import json
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from transformers import AutoTokenizer
from langchain_core.runnables import RunnableLambda
from loguru import logger
import os
import sys
from ml_collections import ConfigDict
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
# from sklearn.metrics.pairwise import cosine_similarity

# 모듈 검색 경로에 /workspace/youngwoo/Agent_m2m/Pages 추가
sys.path.append(os.path.abspath("/workspace/youngwoo/Agent_m2m/Pages"))

class QueryDataFrame:
    def __init__(self, db_path, openai_api_key):
        self.db_path = db_path
        self.llm = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo', api_key=openai_api_key)
        self.db = SQLDatabase.from_uri(f"sqlite:///{self.db_path}")

        self.schema = self.db.get_table_info()
        self.context = self._load_schema_info()
        self.query_examples = self._load_query_examples()  # Load query examples

        self.embeddings = OpenAIEmbeddings()

        self.example_selector = self._initialize_example_selector(self.query_examples)  # Initialize example selector
        self.tag_selector = self._initialize_tag_selector(["전압", "전력", "효율성", "손실", "배기가스", "연료", "압력", "온도", "상태"])  # Initialize tag selector

        self.make_sql_query_chain = (RunnableLambda(self._make_sql_query_prompt) | self.llm | StrOutputParser())

    def _load_schema_info(self):
        json_file_path = "/workspace/youngwoo/Agent_m2m/Pages/dataset/schema_info.json"
        with open(json_file_path, 'r', encoding='utf-8') as json_file:
            schema_info = json.load(json_file)
            return schema_info["table_info"]
        
    def _load_query_examples(self):
        json_file_path = "/workspace/youngwoo/Agent_m2m/Pages/dataset/query_example.json"
        with open(json_file_path, 'r', encoding='utf-8') as json_file:
            query_examples = json.load(json_file)
            return query_examples

    def _initialize_example_selector(self, examples):
        example_selector = SemanticSimilarityExampleSelector.from_examples(
            examples,
            OpenAIEmbeddings(),
            FAISS,
            k=2,
            input_keys=["input"],
        )
        return example_selector

    def _initialize_tag_selector(self, tags):
        tag_examples = [{"input": tag} for tag in tags]
        tag_selector = SemanticSimilarityExampleSelector.from_examples(
            tag_examples,
            OpenAIEmbeddings(),
            FAISS,
            k=1,
            input_keys=["input"],
        )
        return tag_selector

    def _make_sql_query_prompt(self, state):
        # Select the most similar examples based on user query
        selected_examples = self.example_selector.select_examples({"input": state["query"]})
        examples_str = "\n".join([f"사용자 질문: {ex['input']}\nSQLQuery: ```sql\n{ex['query']}\n```" for ex in selected_examples])
        
        # prompt_template = '''
        # 당신은 SQLite 전문가입니다. 주어진 질문을 기반으로 가장 먼저 실행해야 하는 '구문이 올바른 SQLite 쿼리'를 생성하세요.
        # 데이터베이스에서 가장 유익한 데이터를 반환하도록 결과를 정렬할 수 있습니다. 질문에 답변하는 데 필요한 열만 쿼리해야 합니다. 
        # 각 열 이름을 이스케이프 식별자로 나타내기 위해 큰따옴표(")로 감싸십시오. 
        # 아래 테이블에서 볼 수 있는 열 이름만 사용하십시오. 존재하지 않는 열에 대해 쿼리하지 않도록 주의하십시오. 
        # 또한 각 열이 어떤 테이블에 있는지 주의하십시오. 질문에 "오늘"이 포함된 경우 현재 날짜를 가져오기 위해 date('now') 함수를 사용하십시오.
        # 정확한 날짜를 모를 경우, '%'를 사용하세요.
        
        # 사용자 질문: {query}

        # - 생성할 SQLQuery 예시 :
        # {examples}

        # - 다음 형식을 사용하십시오:
        # SQLQuery:  ```sql\n<sql query>\n```

        # - 아래의 column만 사용하십시오:
        # {tags}
        # '''
        # prompt_template = '''
        # You are an agent designed to interact with a SQL database.
        # Given an input question, create a syntactically correct SQLite query to run, then look at the results of the query and return the answer.
        # Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 2 results.
        # You can order the results by a relevant column to return the most interesting examples in the database.
        # Never query for all the columns from a specific table, only ask for the relevant columns given the question.
        # You have access to tools for interacting with the database.
        # Only use the given tools. Only use the information returned by the tools to construct your final answer.
        # You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

        # DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

        # If the question does not seem related to the database, just return "I don't know" as the answer.

        # Here are some examples of user inputs and their corresponding SQL queries:
        # {examples}

        # - Question : {query}

        # - Use the following format:
        # SQLQuery:  ```sql\n<sql query>\n```

        # - Only use the columns below:
        # {tags}
        # '''
        prompt_template = '''
        You are an agent designed to interact with a SQL database.
        Given an input question, create a syntactically correct SQLite query to run, then look at the results of the query and return the answer.
        You can order the results by a relevant column to return the most interesting examples in the database.
        Never query for all the columns from a specific table, only ask for the relevant columns given the question.
        You have access to tools for interacting with the database.
        Only use the given tools. Only use the information returned by the tools to construct your final answer.
        You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

        DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

        If the question does not seem related to the database, just return "I don't know" as the answer.

        Here are some examples of user inputs and their corresponding SQL queries:
        {examples}

        - Question : {query}
        
        - Use the following format:
        SQLQuery:  ```sql\n<sql query>\n```

        - Only use the columns below:
        {tags}
        '''

        # matching_tags를 문자열로 변환
        tags_str = ", ".join(state["matching_tags"])

        prompt = prompt_template.format(query=state["query"], tags=tags_str, examples=examples_str)
        print('-----------------------------------------------------------------------------')
        print(prompt)
        print('-----------------------------------------------------------------------------')
        return prompt

    def query_dataframe(self, user_question, info_df):
        # # Find the most similar category using the tag selector
        # question_embedding = self.embeddings.embed(user_question)
        # tag_embeddings = [self.embeddings.embed(tag) for tag in self.tags]

        # # Compute cosine similarity between the user question and each tag
        # similarities = cosine_similarity([question_embedding], tag_embeddings).flatten()

        # # Find the most similar tag and its similarity score
        # most_similar_index = np.argmax(similarities)
        # most_similar_tag = self.tags[most_similar_index]
        # most_similar_score = similarities[most_similar_index]

        # # Log the most similar tag and its similarity score
        # print(f"Most similar tag: {most_similar_tag} with similarity score: {most_similar_score}")


        selected_tag_example = self.tag_selector.select_examples({"input": user_question})
        most_similar_tag = selected_tag_example[0]["input"]

        # Find the corresponding tags for the most similar category in info_df
        matching_tags = info_df[info_df["센서 분류"] == most_similar_tag]["센서 태그명"].tolist()
        matching_tags.insert(0, "datetime")

        # matching_tags를 문자열로 변환하여 출력
        print("매칭된 태그:", matching_tags)

        # Invoke the LLM with the additional tag information
        tmp = self.make_sql_query_chain.invoke({"query": user_question, "context": self.context, "matching_tags": matching_tags}).strip()

        # Extract the SQL query from the response
        start_idx = tmp.find("```sql") + len("```sql")
        end_idx = tmp.find("```", start_idx)
        sql_query = tmp[start_idx:end_idx].strip()

        # Log the extracted SQL query for debugging
        logger.debug(f"Extracted SQL query: {sql_query}")

        # Connect to the SQLite database
        conn = sqlite3.connect(self.db_path)

        # Execute the SQL query and save the result to a Pandas DataFrame
        data_df = pd.read_sql_query(sql_query, conn)

        # Close the database connection
        conn.close()
        # datetime 체크
        if 'datetime' in data_df.columns:
            data_df['datetime'] = pd.to_datetime(data_df['datetime'], errors='coerce')

        return data_df