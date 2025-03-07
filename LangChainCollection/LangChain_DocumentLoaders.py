# Adapted from "LangChain Chat with Your Data" from "DeepLearning.AI".

# Install necessary dependencies (uncomment and run if needed)
# !pip install --upgrade langchain
# !pip install pypdf
# !pip install yt_dlp  # For downloading YouTube audio
# !pip install -U langchain-community  # Community-contributed utilities
# !pip install pydub  # For handling audio files

import os
import openai
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set OpenAI API key for authentication with OpenAI services
os.environ['OPENAI_API_KEY'] = ""  # Fill in with your OpenAI API key: https://platform.openai.com/api-keys
openai.api_key = os.environ['OPENAI_API_KEY']

# Import necessary LangChain modules for document loading and parsing
from langchain_community.document_loaders.generic import GenericLoader, FileSystemBlobLoader
from langchain.document_loaders.parsers import OpenAIWhisperParser
from langchain.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader

# Define YouTube video URL and directory to save audio files
url = "https://www.youtube.com/watch?v=jGwO_UgTS7I"
save_dir = "docs/youtube/"

# Create a document loader for extracting content from a YouTube video
loader = GenericLoader(
    YoutubeAudioLoader([url], save_dir),  # Downloads audio from YouTube
    # FileSystemBlobLoader(save_dir, glob="*.m4a"),  # Uncomment to load locally saved audio files
    OpenAIWhisperParser()  # Uses OpenAI Whisper to transcribe audio
)

# Load and process the documents
docs = loader.load()

# Print the first 500 characters of the transcribed text
print(docs[0].page_content[0:500])

# Import WebBaseLoader for loading web pages as documents
from langchain.document_loaders import WebBaseLoader

# Load text content from a GitHub-hosted document
loader = WebBaseLoader("https://github.com/basecamp/handbook/blob/master/titles-for-programmers.md")
docs = loader.load()

# Print a snippet of the loaded web content (from character 500 to 1000)
print(docs[0].page_content[500:1000])

# Import NotionDirectoryLoader for loading Notion database content
from langchain.document_loaders import NotionDirectoryLoader

# Load Notion documents from a specified local directory
loader = NotionDirectoryLoader("docs/Notion_DB")
docs = loader.load()

# Print the first 200 characters of the Notion document
print(docs[0].page_content[0:200])

# Display metadata of the first loaded Notion document
docs[0].metadata
