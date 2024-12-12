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
            "en": '''Answer the following questions as best you can. You have access to the following tools:

            {tools}

            Use the following format:

            Question: the input question you must answer
            Thought: you should always think about what to do
            Action: the action to take, should be one of [{tool_names}]
            Action Input: the input to the action
            Observation: the result of the action
            ... (this Thought/Action/Action Input/Observation can repeat N times)
            Thought: I now know the final answer
            Final Answer: the final answer to the original input question

            Begin!

            Question: {input}
            Thought:{agent_scratchpad}''',


            "ko": """
            다음 질문에 최선을 다해 답변하세요. 다음 도구들을 사용할 수 있습니다:

            {tools}

            다음 형식을 사용하세요:
            Question: 답변해야 할 입력 질문
            Thought: 무엇을 해야 할지 항상 생각해야 합니다
            Action: 취해야 할 행동, [{tool_names}] 중 하나여야 합니다
            Action Input: 행동에 대한 입력
            Observation: 행동의 결과
            ... (이 생각/행동/행동 입력/관찰이 여러 번 반복될 수 있습니다)
            Thought: 이제 최종 답을 알겠습니다.
            Final Answer:
            {{
                "output": "입력 질문에 대한 최종 한국어 답변"
            }}

            시작하세요!

            Question: {input}
            Thought: {agent_scratchpad}
                """,




            "gpt_en": """
                **INSTRUCTIONS**
                Write Python pandas code and SQLite query to obtain the answer to the user's question.

                If necessary, use the tools provided below:
                {tools}
                        
                Refer to the tools as follows: {tool_names}

                Final Answer: 
                {{
                    "sql": "Generated SQLite query",
                    "pandas": "Generated Python pandas code"
                }}
                **CONTEXT**
                Each column's value :
                {context_df} 
                Each column's type :
                {df_type}
                Note that the datetime column is in 'string format' with 1-second intervals, e.g., '2023-03-24 00:00:00'
                
                **QUESTION**
                Question: {input} 
                Thought: {agent_scratchpad}
                Action:
                """,

            "en": """

                **INSTRUCTIONS**
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
                Each column's value :
                {context_df} 
                Each column's type :
                {df_type}
                Note that the datetime column is in 'string format' with 1-second intervals, e.g., '2023-03-24 00:00:00'
                
                **QUESTION**

                Question: {input} 
                Thought: {agent_scratchpad}
                Action:
                """,

            "default": """
            Answer the following questions as best you can. You have access to the following tools:

            {tools}

            Use the following format:

            Question: the input question you must answer
            Thought: you should always think about what to do
            Action: the action to take, should be one of [{tool_names}]
            Action Input: the input to the action
            Observation: the result of the action
            ... (this Thought/Action/Action Input/Observation can repeat N times)
            Thought: I now know the final answer
            Final Answer: the final answer to the original input question

            Begin!

            Question: {input}
            Thought:{agent_scratchpad}""",
            
            }

    def get_prompt(self, lang):
        return self.prompts.get(lang, "")


'''Answer the following questions as best you can. You have access to the following tools:

            {tools}

            Use the following format:

            Question: the input question you must answer
            Thought: you should always think about what to do
            Action: the action to take, should be one of [{tool_names}]
            Action Input: the input to the action
            Observation: the result of the action
            ... (this Thought/Action/Action Input/Observation can repeat N times)
            Thought: I now know the final answer
            Final Answer: the final answer to the original input question

            Begin!

            Question: {input}
            Thought:{agent_scratchpad}'''


'''
            "ko": """
                **INSTRUCTIONS**
                사용자가 요청한 데이터셋을 분석하라는 지시를 받았습니다. 

                도구를 사용하여 사용자 요청에 답변하고 한국어로 답변하세요:
                {tools}
                        
                도구를 다음과 같이 참조하세요: {tool_names}

                다음 형식을 사용하세요:

                Question: 답변해야 하는 입력 질문
                Thought: 어떤 조치를 취해야 하는지에 대한 생각
                Action: 원하는 결과를 얻기 위해 사용될 도구를 명시
                Action Input: 도구에 대한 입력
                Observation: 조치의 결과와 의미
                ... (이 생각/행동/행동 입력/관찰이 N번 반복될 수 있음)
                Thought: 이제 최종 답을 알겠습니다
                Final Answer: 
                {{
                    "output":"사용자의 질문에 대한 최종 답변",
                    "pandas":"질문과 관련된 파이썬 pandas 코드"
                }}
                **CONTEXT**
                여기에 df.head()가 있습니다:
                {context_df}

                **QUESTION**

                Question: {input} 
                Thought: {agent_scratchpad}
                Action:
                """            }
'''