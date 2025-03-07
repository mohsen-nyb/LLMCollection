# Adapted from "LangChain for LLM Application Development" from DeepLearning.AI.

# Install necessary dependencies if not already installed
# !pip install -U langchain-community
# !pip install docarray
# !pip install tiktoken
# !pip install langchain_experimental

import os
import openai
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set OpenAI API key for authentication with OpenAI services
os.environ['OPENAI_API_KEY'] = ""  # Fill in with your OpenAI API key: https://platform.openai.com/api-keys
openai.api_key = os.environ['OPENAI_API_KEY']

# Importing experimental LangChain modules for advanced tool and agent functionalities
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain_experimental.agents.agent_toolkits import create_python_agent
from langchain_experimental.utilities import PythonREPL

# Import standard LangChain components for tool usage and agent initialization
from langchain.agents import load_tools, initialize_agent
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI

# Initialize a language model with a temperature of 0 for deterministic responses
llm = ChatOpenAI(temperature=0)

# Load built-in tools: "llm-math" (for math calculations) and "wikipedia" (for knowledge retrieval)
tools = load_tools(["llm-math", "wikipedia"], llm=llm)

# Create an agent that utilizes the loaded tools with a reasoning mechanism (Zero-shot React)
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,  # Uses reasoning to answer questions
    handle_parsing_errors=True,  # Prevents the agent from crashing due to parsing errors
    verbose=True  # Enables detailed logs for debugging
)

# Example usage: Asking the agent a math-related question
agent("what is 25 percent of 100?")

# Example usage: Asking a Wikipedia-related question
question = "Tom M. Mitchell is an American computer scientist \
and the Founders University Professor at Carnegie Mellon University (CMU)\
what book did he write?"
result = agent(question)

# Creating an agent that can execute Python code using the Python REPL tool
agent = create_python_agent(
    llm, 
    tool=PythonREPLTool(),  # Allows execution of Python code
    verbose=True
)

# Example task: Sorting a list of customers by last name and then first name
customer_list = [["Harrison", "Chase"], 
                 ["Lang", "Chain"],
                 ["Dolly", "Too"],
                 ["Elle", "Elem"], 
                 ["Geoff","Fusion"], 
                 ["Trance","Former"],
                 ["Jen","Ayai"]
                ]
agent.run(f"""Sort these customers by \
last name and then first name \
and print the output: {customer_list}""") 

# Enable debugging mode to get more details on execution
import langchain
langchain.debug = True

# Run the same sorting task with debugging enabled
agent.run(f"""Sort these customers by \
last name and then first name \
and print the output: {customer_list}""") 

# Disable debugging mode
langchain.debug = False

# Building a custom tool to return the current date
from langchain.agents import tool
from datetime import date

@tool
def time(text: str) -> str:
    """Returns today's date. Use this tool for any \
    questions related to knowing today's date. \
    The input should always be an empty string, \
    and this function will always return today's \
    date - any date calculations should occur \
    outside this function."""
    return str(date.today())

# Adding the custom time tool to the existing agent
agent = initialize_agent(
    tools + [time],  # Append the time tool to the list of tools
    llm,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose=True
)

# Example usage: Asking the agent for the current date
try:
    result = agent("what's the date today?") 
except: 
    print("Exception on external access")
