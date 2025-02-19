#adapted from "LangChain for LLM Application Development" from "DeepLearning.AI".

#!pip install --upgrade langchain
#!pip install -U langchain-community
#!pip install tiktoken

from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory, ConversationSummaryBufferMemory
import os
import openai

# Set your OpenAI API key
os.environ['OPENAI_API_KEY'] = ""  # Fill in with your OpenAI API key from: https://platform.openai.com/api-keys
openai.api_key = os.environ['OPENAI_API_KEY']
llm_model = "gpt-3.5-turbo"

# 1. ConversationBufferMemory
llm = ChatOpenAI(temperature=0.0, model=llm_model)

# Initialize buffer memory to store the entire conversation history
memory = ConversationBufferMemory()
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

# Start a conversation
conversation.predict(input="Hey, my name is Mohsen!")
conversation.predict(input="Can you help me plan my wedding?")

# Display full conversation buffer
print(memory.buffer)

# Load memory variables
print(memory.load_memory_variables({}))

# Manually add to memory
memory.save_context({"input": "Hi"}, {"output": "What's up"})
print(memory.buffer)


# 2. ConversationBufferWindowMemory
# Stores only the last 'k' interactions for short-term context
memory = ConversationBufferWindowMemory(k=1)

# Save interactions
memory.save_context({"input": "Hi"}, {"output": "What's up"})
memory.save_context({"input": "Not much, just hanging"}, {"output": "Cool"})

# Load the last 'k' interactions
print(memory.load_memory_variables({}))


# 3. ConversationSummaryBufferMemory
# Summarizes long conversations while keeping context under token limits

# Example long schedule to test summarization
schedule = "There is a meeting at 8am with your product team. " \
           "You will need your PowerPoint presentation prepared. " \
           "9am-12pm have time to work on your LangChain project which will go quickly " \
           "because LangChain is such a powerful tool. At Noon, lunch at the Italian restaurant " \
           "with a customer who is driving over an hour to meet you to understand the latest in AI. " \
           "Be sure to bring your laptop to show the latest LLM demo."

# Initialize summarizing memory with a token limit
memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=100)

# Add context to the memory
memory.save_context({"input": "Hello"}, {"output": "What's up"})
memory.save_context({"input": "Not much, just hanging"}, {"output": "Cool"})
memory.save_context({"input": "What is on the schedule today?"}, {"output": f"{schedule}"})

# Display summarized memory buffer
print(memory.buffer)
