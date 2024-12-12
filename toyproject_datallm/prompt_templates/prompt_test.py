# sql_prompt = """
# 주어진 질문을 바탕으로, 먼저 실행할 구문적으로 올바른 SQLite 쿼리를 작성하세요.
# 다음 형식을 사용하세요:

# Question: "질문 내용"
# SQLQuery: "실행할 SQL 쿼리"

# Question: {question}
# """


# 만약 agent로 만든 후 답변 출력이 안되는 모델이 있으면 추가.
# These keywords must never be translated and transformed:
# - Action:
# - Thought:
# - Action Input:

class Prompt:
    def __init__(self):
        self.prompts = {
            "gpt_en": """
                **INSTRUCTIONS**
                Write Python pandas code to obtain the answer to the user's question.

                If necessary, use the tools provided below:
                {tools}
                        
                Refer to the tools as follows: {tool_names}

                Final Answer: 
                {{
                    "sql" : "Generated SQL Query"
                }}
                **CONTEXT**
                Each column's type :
                {df_type}
                Note that the datetime column is in 'string format' with 1-second intervals, e.g., '2023-03-24 00:00:00'
                
                **QUESTION**
                Question: {input} 
                Thought: {agent_scratchpad}
                Action:
                """
                }
    def get_prompt(self, lang):
        return self.prompts.get(lang, "")