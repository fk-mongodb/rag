from pymongo import MongoClient
from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch
from pymongo import MongoClient
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

import os

# The dotenv library is used to load environment variables from a .env file
load_dotenv() 

# initialize MongoDB python client
client = MongoClient(os.environ['MONGODB_URI'])

DB_NAME = os.environ['MONGODB_DB']
COLLECTION_NAME = os.environ['MONGODB_VECTOR_COLL_LANGCHAIN']

MONGODB_COLLECTION = client[DB_NAME][COLLECTION_NAME]

vector_store = MongoDBAtlasVectorSearch(
    collection=MONGODB_COLLECTION,
    embedding=OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY']),
    index_name=os.environ['MONGODB_VECTOR_INDEX'],
    relevance_score_fn="cosine",
)

results = vector_store.similarity_search_with_score(query="Horror",k=1)
for doc, score in results:
    print(f"* [SIM={score:3f}] {doc.page_content} [{doc.metadata}]")

# results = vector_store.similarity_search(query="Pauline",k=1)
# for doc in results:
#     print(f"* {doc.page_content} [{doc.metadata}]")