from pymongo import MongoClient
from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch
from pymongo import MongoClient
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

import os

# The dotenv library is used to load environment variables from a .env file
load_dotenv() 

# initialize MongoDB python client
client = MongoClient(os.environ['MONGODB_URI'])

DB_NAME = os.environ['MONGODB_DB']
COLLECTION_NAME = os.environ['MONGODB_VECTOR_COLL_LANGCHAIN']

MONGODB_COLLECTION = client[DB_NAME][COLLECTION_NAME]

vector_search = MongoDBAtlasVectorSearch(
    collection=MONGODB_COLLECTION,
    embedding=OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY']),
    index_name=os.environ['MONGODB_VECTOR_INDEX'],
    relevance_score_fn="cosine",
)

# perform a similarity search on the ingested documents
prompt='What is the best horror movie to watch?'
docs_with_score = vector_search.similarity_search_with_score(query=prompt,k=1)

llm = ChatOpenAI(openai_api_key=os.environ['OPENAI_API_KEY'])

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a movie recommendation engine which posts a concise and short summary on relevant movies."),
    ("user", "List of movies: {input}")
])

# Create an LLMChain
chain = LLMChain(
    llm=llm,
    prompt=prompt_template
)

# Prepare the input for the chat model
input_docs = "\n".join([doc.page_content for doc, _ in docs_with_score])

# Invoke the chain with the input documents
response = chain.invoke({"input": input_docs})
print(response['text'])