from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_experimental.utilities import PythonREPL

template = """
Write some python  code to solve the user's problem. 
Include print(result) at the end for user to check the result. 
Return only python code and nothing else."""

prompt = ChatPromptTemplate.from_messages(
    [("system", template), ("human", "{question}")]
)

model = ChatOpenAI(temperature=0, model_name="gpt-4")

#  Python codes generation
PythonCode_chain = prompt | model | StrOutputParser()

# Python REPL (Read-Eval-Print Loop)
PythonCodeRun_chain = PythonCode_chain | PythonREPL().run