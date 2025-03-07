# Adapted from "LangChain Chat with Your Data" from "DeepLearning.AI".

# Install necessary dependencies (uncomment and run if needed)
# !pip install --upgrade langchain
# !pip install pypdf
# !pip install yt_dlp  # For downloading YouTube audio
# !pip install -U langchain-community  # Community-contributed utilities
# !pip install pydub  # For handling audio files
# !pip install -qU langchain-text-splitters
# !pip install tiktoken 

import os
import openai
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set OpenAI API key for authentication with OpenAI services
os.environ['OPENAI_API_KEY'] = ""  # Fill in with your OpenAI API key: https://platform.openai.com/api-keys
openai.api_key = os.environ['OPENAI_API_KEY']

# Importing text splitting utilities from LangChain
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter, TokenTextSplitter
from langchain.document_loaders import PyPDFLoader

# Define chunk size and overlap for text splitting
chunk_size = 26
chunk_overlap = 4

# Initialize text splitters
r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap
)
c_splitter = CharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap
)

# Example text to demonstrate splitting
text1 = 'abcdefghijklmnopqrstuvwxyz'
print(r_splitter.split_text(text1))
print()

text2 = 'abcdefghijklmnopqrstuvwxyzabcdefg'
print(r_splitter.split_text(text2))
print(c_splitter.split_text(text2))
print()

text3 = "a b c d e f g h i j k l m n o p q r s t u v w x y z"
print(r_splitter.split_text(text3))
print(c_splitter.split_text(text3))
print()

# Custom separator for space-based splitting
c_splitter = CharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    separator=' '
)
c_splitter.split_text(text3)

# More complex text example
document_text = """When writing documents, writers will use document structure to group content. \
This can convey to the reader, which ideas are related. For example, closely related ideas \
are in sentences. Similar ideas are in paragraphs. Paragraphs form a document.\n\n  \
Paragraphs are often delimited with a carriage return or two carriage returns. \
Carriage returns are the "backslash n" you see embedded in this string. \
Sentences have a period at the end, but also, have a space.\
and words are separated by space."""

# Splitting with different strategies
c_splitter = CharacterTextSplitter(
    chunk_size=450,
    chunk_overlap=0,
    separator=' '
)
r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=450,
    chunk_overlap=0, 
    separators=["\n\n", "\n", " ", ""]
)

print(c_splitter.split_text(document_text))
print()
print(r_splitter.split_text(document_text))

# Recursive splitting with more granularity
r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=0,
    separators=["\n\n", "\n", "\. ", " ", ""]
)
r_splitter.split_text(document_text)

r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=0,
    separators=["\n\n", "\n", "(?<=\. )", " ", ""]
)
r_splitter.split_text(document_text)

# Loading and splitting a PDF document
loader = PyPDFLoader("MachineLearning-Lecture01.pdf")
pages = loader.load()

# Token-based splitting
text_splitter = TokenTextSplitter(chunk_size=1, chunk_overlap=0)
text1 = "foo bar bazzyfoo"
text_splitter.split_text(text1)

text_splitter = TokenTextSplitter(chunk_size=2, chunk_overlap=0)
text_splitter.split_text(text1)

# Context-aware Markdown splitting
from langchain.text_splitter import MarkdownHeaderTextSplitter

# Example markdown text
markdown_document = """# Title\n\n \
## Chapter 1\n\n \
Hi this is Jim\n\n Hi this is Joe\n\n \
### Section \n\n \
Hi this is Lance \n\n \
## Chapter 2\n\n \
Hi this is Molly"""

# Define headers to split on
headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

# Initialize and apply Markdown splitter
markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on
)
md_header_splits = markdown_splitter.split_text(markdown_document)
