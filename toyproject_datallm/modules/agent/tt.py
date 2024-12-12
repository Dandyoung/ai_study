import os
import sqlite3
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain.chains.sql_database.query import create_sql_query_chain
from langchain_community.utilities import SQLDatabase
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_experimental.utilities import PythonREPL
from langchain.agents.agent_types import AgentType
from langchain.agents import AgentExecutor, Tool
from langchain.prompts import PromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.agents import create_react_agent
from prompt_templates.prompt_test import Prompt
from langchain_experimental.agents.agent_toolkits import create_csv_agent

# Helper functions
def get_schema(_):
    return db.get_table_info()

def run_query(query):
    return db.run(query)

# Initialize the database connection
db = SQLDatabase.from_uri("sqlite:////workspace/youngwoo/toyproject-datallm/modules/agent/gt_v1_head.db")

# Initialize the language model
model = ChatOpenAI(temperature=0, model_name='gpt-4', api_key=os.getenv("OPENAI_API_KEY"))

# Prompt for generating a SQL query
template_query = """
Based on the table schema below, 
Write a PostgreSQL query that answer the user's question:
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
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(query)
    results = cursor.fetchall()
    columns = [description[0] for description in cursor.description]
    conn.close()
    return columns, results

# Initialize the agent
def agent_init(_llm, path, language):
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
                       ),
        ResponseSchema(
            name="pandas",
            description="Write Python code related to the question using pandas."
        )
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    df = pd.read_csv(path)

    context_df = df.head().to_string()
    tpye_df = df.dtypes

    prompt_instance = Prompt()
    agent_prompt = prompt_instance.get_prompt(language)
    
    system_prompt = "You are a skilled data analyst.\n Please ensure that the query and answer exist, and improve it into a graph that can alleviate the answer.\n"
    combined_prompt = system_prompt + agent_prompt
    prompt_template = PromptTemplate.from_template(combined_prompt)

    agent = create_react_agent(llm=_llm, tools=tools, prompt=prompt_template)
    agent_chain = AgentExecutor.from_agent_and_tools(agent=agent, 
                                                     tools=tools, 
                                                     handle_parsing_errors=True, 
                                                     max_execution_time=60, 
                                                     max_iterations=10, 
                                                     verbose=True)

    return agent_chain, output_parser, df, context_df, tools, tool_names , tpye_df

# Function to process data in chunks
def process_data_in_chunks(db_path, question, chunk_size=3000):
    sql_query = generate_sql_query(question)
    print("Generated SQL Query:", sql_query)

    columns, results = execute_sql_query(db_path, sql_query)
    all_results = results.copy()

    while len(results) == chunk_size:
        sql_query = update_sql_query_for_next_chunk(sql_query, columns, results[-1])
        columns, results = execute_sql_query(db_path, sql_query)
        all_results.extend(results)

    return columns, all_results

def update_sql_query_for_next_chunk(sql_query, columns, last_row):
    # Update the SQL query to fetch the next chunk of data based on the last_row from the previous results
    # This is a placeholder and should be customized based on the specific query and database schema
    return sql_query

# Main function to process chunks with the agent and summarize results
def main():
    llm = ChatOpenAI(temperature=0.1,
                    model="gpt-4-turbo", 
                    verbose=True, 
                    api_key=os.getenv("OPENAI_API_KEY"))
    agent_chain, output_parser, df, context_df, tools, tool_names, df_type = agent_init(llm, 'gt_v1_head.csv', language='gpt_en')

    prefix = "You are a skilled data analyst.\n"
    question = "Tell me the data from 12 PM to 3 PM on March 24th."
    user_input = prefix + question

    columns, all_results = process_data_in_chunks("/workspace/youngwoo/toyproject-datallm/modules/agent/gt_v1_head.db", question)
    chunk_size = 3000
    all_responses = []

    analysis_prompt_template = """
    **INSTRUCTIONS**
    You are a skilled data analyst. Based on the user's question, analyze the given data.
    User's question: {user_question}

    **DATA**
    {data_chunk}
    """

    for i in range(0, len(all_results), chunk_size):
        chunk = all_results[i:i+chunk_size]
        data_chunk = pd.DataFrame(chunk, columns=columns).to_string()
        analysis_prompt = analysis_prompt_template.format(user_question=question, data_chunk=data_chunk)
        response = agent_chain.invoke({
            "input": analysis_prompt,
            "context_df": context_df,
            "tools": tools,
            "tool_names": tool_names,
            "df_type": df_type
        })
        all_responses.append(response['output'])

    # Combine all responses into one string
    combined_responses = "\n\n".join(all_responses)

    # Summarize the combined responses in chunks
    max_summary_chunk_length = 3000
    summary_chunks = [combined_responses[i:i+max_summary_chunk_length] for i in range(0, len(combined_responses), max_summary_chunk_length)]
    summarized_responses = []

    for chunk in summary_chunks:
        summary_prompt = f"Summarize the following analysis results:\n\n{chunk}"
        summary_response = model({
            "messages": [{"role": "user", "content": summary_prompt}]
        })
        summarized_responses.append(summary_response['choices'][0]['message']['content'])

    # Combine all summarized responses into one final summary
    final_summary_prompt = "Combine and summarize the following analysis summaries:\n\n" + "\n\n".join(summarized_responses)
    final_summary_response = model({
        "messages": [{"role": "user", "content": final_summary_prompt}]
    })

    print("Final Summary:", final_summary_response['choices'][0]['message']['content'])

if __name__ == "__main__":
    main()
