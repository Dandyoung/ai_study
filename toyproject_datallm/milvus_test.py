# from pymilvus import connections
# connections.connect(
#   alias="default", 
#   host='milvus-standalone', 
#   port='19530'
# )
# print("connected...")
import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Milvus
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection

# 설정
COLLECTION_NAME = 'doc_qa_db'
DIMENSION = 768
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
URI = 'http://milvus-standalone:19530'

from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Milvus
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = WebBaseLoader([
    'https://milvus.io/docs/overview.md',
    'https://milvus.io/docs/release_notes.md',
    'https://milvus.io/docs/architecture_overview.md',
    'https://milvus.io/docs/four_layers.md',
    'https://milvus.io/docs/main_components.md',
    'https://milvus.io/docs/data_processing.md',
    'https://milvus.io/docs/bitset.md',
    'https://milvus.io/docs/boolean.md',
    'https://milvus.io/docs/consistency.md',
    'https://milvus.io/docs/coordinator_ha.md',
    'https://milvus.io/docs/replica.md',
    'https://milvus.io/docs/knowhere.md',
    'https://milvus.io/docs/schema.md',
    'https://milvus.io/docs/dynamic_schema.md',
    'https://milvus.io/docs/json_data_type.md',
    'https://milvus.io/docs/metric.md',
    'https://milvus.io/docs/partition_key.md',
    'https://milvus.io/docs/multi_tenancy.md',
    'https://milvus.io/docs/timestamp.md',
    'https://milvus.io/docs/users_and_roles.md',
    'https://milvus.io/docs/index.md',
    'https://milvus.io/docs/disk_index.md',
    'https://milvus.io/docs/scalar_index.md',
    'https://milvus.io/docs/performance_faq.md',
    'https://milvus.io/docs/product_faq.md',
    'https://milvus.io/docs/operational_faq.md',
    'https://milvus.io/docs/troubleshooting.md',
])

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=0)
all_splits = text_splitter.split_documents(docs)


embeddings = OpenAIEmbeddings()
connection_args = { 'uri': URI }

vector_store = Milvus(
    embedding_function=embeddings,
    connection_args=connection_args,
    collection_name=COLLECTION_NAME,
    drop_old=True,
).from_documents(
    all_splits,
    embedding=embeddings,
    collection_name=COLLECTION_NAME,
    connection_args=connection_args,
)

query = "What are the main components of Milvus?"
docs = vector_store.similarity_search(query)

print(len(docs))


llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0) 
retriever = vector_store.as_retriever()

template = """다음 정보를 사용하여 끝에 나오는 질문에 답하세요.
모르겠다면 답을 지어내지 말고 모른다고 말하세요.
세 문장 이내로 최대한 간결하게 답하세요.
답변 끝에는 항상 "질문해 주셔서 감사합니다!"라고 말하세요.
{context}
질문: {question}
유용한 답변:"""
rag_prompt = PromptTemplate.from_template(template)

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
)

print(rag_chain.invoke("밀버스 각기 다른 버전들의 차이점에 대해 설명해줘."))