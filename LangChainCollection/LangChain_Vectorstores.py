# Adapted from "LangChain Chat with Your Data" from "DeepLearning.AI".

# Install necessary dependencies (uncomment and run if needed)
# !pip install -qU langchain-text-splitters
# !pip install --upgrade langchain
# !pip install pypdf
# !pip install yt_dlp  # For downloading YouTube audio
# !pip install -U langchain-community  # Community-contributed utilities
# !pip install tiktoken 
# !pip install chromadb

import os
import openai
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set OpenAI API key for authentication with OpenAI services
os.environ['OPENAI_API_KEY'] = ""  # Fill in with your OpenAI API key: https://platform.openai.com/api-keys
openai.api_key = os.environ['OPENAI_API_KEY']

# Import necessary modules from LangChain
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
import numpy as np

# Load PDF documents (duplicate documents included to simulate messy data)
loaders = [
    PyPDFLoader("docs/MachineLearning-Lecture01.pdf"),
    PyPDFLoader("docs/MachineLearning-Lecture01.pdf"),  # Intentional duplicate
    PyPDFLoader("docs/MachineLearning-Lecture02.pdf"),
    PyPDFLoader("docs/MachineLearning-Lecture03.pdf")
]

# Extract text from PDF files
docs = []
for loader in loaders:
    docs.extend(loader.load())

# Initialize text splitter with chunk size and overlap
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=150
)

# Split the loaded documents into smaller text chunks
splits = text_splitter.split_documents(docs)

# Initialize OpenAI embedding model
embedding = OpenAIEmbeddings()

# Example sentences to compute similarity
sentence1 = "i like dogs"
sentence2 = "i like canines"
sentence3 = "the weather is ugly outside"

# Compute embeddings for each sentence
embedding1 = embedding.embed_query(sentence1)
embedding2 = embedding.embed_query(sentence2)
embedding3 = embedding.embed_query(sentence3)

# Compute cosine similarity between sentence embeddings
similarity = np.dot(embedding1, embedding2)

# Set up persistent vector database storage
persist_directory = 'docs/chroma/'

# Create a vector database from the document embeddings
vectordb = Chroma.from_documents(
    documents=splits,
    embedding=embedding,
    persist_directory=persist_directory
)

# Print the number of stored documents
print(vectordb._collection.count())

# Perform a similarity search on the vector database
question = "is there an email i can ask for help"
queried_docs = vectordb.similarity_search(question, k=3)

# Persist the database to disk for future use
vectordb.persist()
