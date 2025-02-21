#adapted from "LangChain for LLM Application Development" from "DeepLearning.AI".

#!pip install -U langchain-community
#!pip install docarray
#!pip install tiktoken


import os
import openai
import warnings

from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.vectorstores import DocArrayInMemorySearch
from IPython.display import display, Markdown
from langchain.indexes import VectorstoreIndexCreator
from langchain.embeddings import OpenAIEmbeddings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set your OpenAI API key
os.environ['OPENAI_API_KEY'] = ""  # Fill in with your OpenAI API key
openai.api_key = os.environ['OPENAI_API_KEY']
llm_model = "gpt-3.5-turbo"

# Load CSV data
file = 'OutdoorClothingCatalog_1000.csv'
loader = CSVLoader(file_path=file)
embedding = OpenAIEmbeddings()

# Create vector store index
index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch,
    embedding=embedding
).from_loaders([loader])

# Define query for product information
query = "Please list all your shirts with sun protection in a table in markdown and summarize each one."

# Initialize LLM for querying
llm_replacement_model = OpenAI(temperature=0.7, model='gpt-3.5-turbo-instruct')

# Execute query and display result
response = index.query(query, llm=llm_replacement_model)
display(Markdown(response))

# --- Step-by-Step Approach ---

# Load documents
docs = loader.load()

# Create embeddings and vector store
db = DocArrayInMemorySearch.from_documents(docs, embedding)

# Perform similarity search
similar_query = "Please suggest a shirt with sunblocking"
docs = db.similarity_search(similar_query)

# Prepare documents for LLM
retriever = db.as_retriever()
llm = ChatOpenAI(temperature=0.7, model=llm_model)
qdocs = "".join([doc.page_content for doc in docs])

# Direct LLM query with context
response = llm.call_as_llm(f"{qdocs} Question: Please list all your shirts with sun protection in a table in markdown and summarize each one.")
display(Markdown(response))

# --- Using RetrievalQA Chain ---

qa_stuff = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    verbose=True
)

# Run RetrievalQA with query
response = qa_stuff.run(query)
display(Markdown(response))
