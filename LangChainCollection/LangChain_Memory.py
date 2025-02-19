from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory, ConversationSummaryBufferMemory
import os
import openai

os.environ['OPENAI_API_KEY']= "" # fill in with your openai api aky. you can get it from here: https://platform.openai.com/api-keys
openai.api_key = os.environ['OPENAI_API_KEY']
llm_model = "gpt-3.5-turbo"


# 1.ConversationBufferMemory
llm = ChatOpenAI(temperature=0.0, model=llm_model)
memory = ConversationBufferMemory()
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

conversation.predict(input="Hey, my name is Mohsen!")
    # > Entering new ConversationChain chain...
    # Prompt after formatting:
    # The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.
    
    # Current conversation:
    
    # Human: Hey, my name is Mohsen!
    # AI:
    
    # > Finished chain.
    # Hello Mohsen! It's nice to meet you. How can I assist you today?
conversation.predict(input="can you help me plan my wedding?")
print(memory.buffer)
print(memory.load_memory_variables({})

# saving manually to the memory
memory.save_context({"input": "Hi"}, 
                    {"output": "What's up"})
print(memory.buffer)


# 2. ConversationBufferWindowMemory 
memory = ConversationBufferWindowMemory(k=1)
memory.save_context({"input": "Hi"},
                    {"output": "What's up"})
memory.save_context({"input": "Not much, just hanging"},
                    {"output": "Cool"})
memory.load_memory_variables({})



#3. 

#!pip install tiktoken
# create a long string
schedule = "There is a meeting at 8am with your product team. \
You will need your powerpoint presentation prepared. \
9am-12pm have time to work on your LangChain \
project which will go quickly because Langchain is such a powerful tool. \
At Noon, lunch at the italian resturant with a customer who is driving \
from over an hour away to meet you to understand the latest in AI. \
Be sure to bring your laptop to show the latest LLM demo."

memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=100)
memory.save_context({"input": "Hello"}, {"output": "What's up"})
memory.save_context({"input": "Not much, just hanging"},
                    {"output": "Cool"})
memory.save_context({"input": "What is on the schedule today?"}, 
                    {"output": f"{schedule}"})

print(memory.buffer)
