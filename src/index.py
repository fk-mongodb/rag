from dotenv import load_dotenv
from langchain_community.document_loaders.mongodb import MongodbLoader
from pymongo import MongoClient
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

import os

# The nest_asyncio library enables asyncio event loop nesting
import nest_asyncio
nest_asyncio.apply()

# The dotenv library is used to load environment variables from a .env file
load_dotenv() 

# Use the MongoLoader class to retrieve documents from MongoDB
loader = MongodbLoader(
    connection_string=os.environ['MONGODB_URI'],
    db_name=os.environ['MONGODB_DB'],
    collection_name=os.environ['MONGODB_COLL'],
    filter_criteria={},
    field_names=["title", "plot"]
)

# The load() method of the MongoLoader instance is invoked to fetch documents from MongoDB based on the specified parameters
# and further, we print it
docs = loader.load()
print(len(docs))
docs[0]


# Use the client instance to access a specific collection within the MongoDB database. 
# This collection will hold the embedding data along with the text.
client = MongoClient(os.environ['MONGODB_URI'], appname="devrel.content.langchain_llamaIndex.python")
collection = client.get_database(os.environ['MONGODB_DB']).get_collection(os.environ['MONGODB_VECTOR_COLL_LANGCHAIN'])

# Pass the 'docs' variable, containing documents fetched from MongoDB earlier, to be used for setting up the vector search.
# We use OpenAI API key and create an embedding. It then passes the collection instance to the 'collection' 
# parameter where vector embeddings will be stored
vector_search = MongoDBAtlasVectorSearch.from_documents(
  documents=docs,
  embedding=OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY']),
  collection=collection,
  index_name=os.environ['MONGODB_VECTOR_INDEX'])
  # We also specify the name of the vector index (here, vector_index) 
  # within the collection which will be used for the semantic search

# Perform a similarity search on the ingested documents
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