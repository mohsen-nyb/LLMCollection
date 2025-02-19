# Adapted from "LangChain for LLM Application Development" from "DeepLearning.AI".

#!pip install --upgrade langchain
#!pip install -U langchain-community

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain
from langchain.chains.router import MultiPromptChain
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
import os
import openai
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set OpenAI API key
os.environ['OPENAI_API_KEY'] = ""  # Fill in with your OpenAI API key: https://platform.openai.com/api-keys
openai.api_key = os.environ['OPENAI_API_KEY']

# Define LLM model
llm_model = "gpt-3.5-turbo"

# -------------------------
# 1. LLMChain Example
# -------------------------

# Initialize ChatOpenAI with temperature for creativity
llm = ChatOpenAI(temperature=0.9, model=llm_model)

# Create prompt template for naming a company
prompt = ChatPromptTemplate.from_template(
    "What is the best name to describe a company that makes {product}?"
)

# Build LLMChain with LLM and prompt
chain = LLMChain(llm=llm, prompt=prompt)

# Example product
product = "LLM agent systems"

# Run chain with input
chain.run(product)

# -------------------------------
# 2. SimpleSequentialChain Example
# -------------------------------

# First prompt: generate company name
first_prompt = ChatPromptTemplate.from_template(
    "What is the best name to describe a company that makes {product}?"
)
first_chain = LLMChain(llm=llm, prompt=first_prompt)

# Second prompt: create description
second_prompt = ChatPromptTemplate.from_template(
    "Write a 20 words description for the following company: {company_name}"
)
second_chain = LLMChain(llm=llm, prompt=second_prompt)

# Combine chains sequentially
simple_sequential_chain = SimpleSequentialChain(chains=[first_chain, second_chain], verbose=False)

# Run sequential chain
simple_sequential_chain.run(product)

# -----------------------
# 3. SequentialChain Example
# -----------------------

# First Chain: Translate review to English
first_prompt = ChatPromptTemplate.from_template(
    "Translate the following review to English:\n\n{Review}"
)
chain_one = LLMChain(llm=llm, prompt=first_prompt, output_key='English_review')

# Second Chain: Summarize English review
second_prompt = ChatPromptTemplate.from_template(
    "Can you summarize the following review in one sentence:\n\n{English_review}"
)
second_chain = LLMChain(llm=llm, prompt=second_prompt, output_key='summary')

# Third Chain: Detect language of original review
third_prompt = ChatPromptTemplate.from_template(
    "What language is the following review:\n\n{Review}"
)
third_chain = LLMChain(llm=llm, prompt=third_prompt, output_key='language')

# Fourth Chain: Generate follow-up response
fourth_prompt = ChatPromptTemplate.from_template(
    "Write a follow-up response to the following summary in the specified language:\n\nSummary: {summary}\n\nLanguage: {language}"
)
fourth_chain = LLMChain(llm=llm, prompt=fourth_prompt, output_key='followup_message')

# Combine all chains into SequentialChain
sequential_chain = SequentialChain(
    chains=[chain_one, second_chain, third_chain, fourth_chain],
    input_variables=['Review'],
    output_variables=["English_review", "summary", "followup_message"],
    verbose=True
)

# Example review in French
Review = """J'utilise le chatbot LLM Agent depuis quelques semaines maintenant, et il est tout simplement impressionnant ! Le chatbot comprend mes requêtes avec une précision remarquable et fournit des réponses naturelles et pertinentes. Il me fait gagner beaucoup de temps en automatisant les tâches répétitives et m’aide même à générer des suggestions intéressantes pour mon travail. L'une de ses meilleures fonctionnalités est sa capacité à apprendre et à s’adapter au fil du temps, rendant les interactions plus fluides et efficaces. L'interface est conviviale, et l'intégration avec mes outils existants s'est faite sans problème. Si vous recherchez un assistant IA performant qui améliore la productivité et fournit des réponses fiables, je recommande vivement ce chatbot. Un véritable atout pour les entreprises et les particuliers !"""

# Run SequentialChain
sequential_chain(Review)

# ------------------
# 4. Router Chain
# ------------------

# Define specialized prompts for different domains
physics_template = """You are a very smart physics professor. You are great at answering questions about physics in a concise and easy to understand manner. When you don't know the answer to a question you admit that you don't know.\n\nHere is a question:\n{input}"""

math_template = """You are a very good mathematician. You are great at answering math questions. You are so good because you are able to break down hard problems into their component parts, answer the component parts, and then put them together to answer the broader question.\n\nHere is a question:\n{input}"""

history_template = """You are a very good historian. You have an excellent knowledge of people, events, and contexts from a range of historical periods. You can think, reflect, debate, discuss, and evaluate the past.\n\nHere is a question:\n{input}"""

computerscience_template = """You are a successful computer scientist. You excel in solving complex problems, explaining algorithms, and coding.\n\nHere is a question:\n{input}"""

# Store prompt information
prompt_infos = [
    {"name": "physics", "description": "Good for answering questions about physics", "prompt_template": physics_template},
    {"name": "math", "description": "Good for answering math questions", "prompt_template": math_template},
    {"name": "history", "description": "Good for answering history questions", "prompt_template": history_template},
    {"name": "computer science", "description": "Good for answering computer science questions", "prompt_template": computerscience_template}
]

# Initialize LLM for router
llm = ChatOpenAI(temperature=0, model=llm_model)

# Create destination chains for each domain
destination_chains = {}
for p_info in prompt_infos:
    name = p_info["name"]
    prompt = ChatPromptTemplate.from_template(template=p_info["prompt_template"])
    chain = LLMChain(llm=llm, prompt=prompt)
    destination_chains[name] = chain

# Create default chain for unmatched queries
default_prompt = ChatPromptTemplate.from_template("{input}")
default_chain = LLMChain(llm=llm, prompt=default_prompt)

# Router template for directing inputs
MULTI_PROMPT_ROUTER_TEMPLATE = """Given a raw text input to a language model, select the model prompt best suited for the input.\nYou will be given the names of the available prompts and a description of what each is best suited for.\n\n<< CANDIDATE PROMPTS >>\n{destinations}\n\n<< INPUT >>\n{{input}}\n\n<< OUTPUT >>\n```json\n{{{{\n    \"destination\": \"DEFAULT\" or the name of the prompt,\n    \"next_inputs\": \"Modified or original input\"\n}}}}\n```"""

# Format destination descriptions
destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
destinations_str = "\n".join(destinations)

# Build router prompt
router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(destinations=destinations_str)

# Initialize Router Chain
router_prompt = ChatPromptTemplate.from_template(router_template)
router_chain = LLMRouterChain.from_llm(llm, router_prompt)

# Combine router with destination chains
multi_prompt_chain = MultiPromptChain(
    router_chain=router_chain,
    destination_chains=destination_chains,
    default_chain=default_chain,
    verbose=True
)

# Test Router Chain with different queries
multi_prompt_chain.run("What is black body radiation?")
multi_prompt_chain.run("What is a derivative?")
multi_prompt_chain.run("What is the derivative of acceleration?")
multi_prompt_chain.run("Why does every cell in our body contain DNA?")
