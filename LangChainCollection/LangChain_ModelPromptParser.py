#adapted from "LangChain for LLM Application Development" from "DeepLearning.AI".

#!pip install --upgrade langchain
#!pip install -U langchain-community

import os
import openai
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

# Set up OpenAI API Key
os.environ['OPENAI_API_KEY'] = ""  # Fill in with your OpenAI API key: https://platform.openai.com/api-keys
openai.api_key = os.environ['OPENAI_API_KEY']
llm_model = "gpt-3.5-turbo"

# Initialize ChatOpenAI with temperature control
chat = ChatOpenAI(temperature=0, model=llm_model)  # temperature=0 ensures deterministic responses

# --------------------------
# Example 1: Translation with Prompt Template
# --------------------------

# Define a template for translating text into a specific style
template_string = """Translate the text that is delimited by triple backticks into a style that is {style}.\ntext: ```{text}```"""

# Create a prompt template
prompt_template = ChatPromptTemplate.from_template(template_string)
print(prompt_template.messages[0].prompt.input_variables)  # Should print ['style', 'text']

# Define the desired style and customer email
customer_style = """American English in a calm and respectful tone"""

customer_email = """
Arrr, I be fuming that me blender lid flew off and splattered me kitchen walls with smoothie! And to make matters worse, the warranty don't cover the cost of cleaning up me kitchen. I need yer help right now, matey!
"""

# Format the prompt with style and text
customer_messages = prompt_template.format_messages(style=customer_style, text=customer_email)

# Get LLM response
customer_response = chat(customer_messages)
print(customer_response.content)

# --------------------------
# Example 2: Extracting Structured Data
# --------------------------

# Define customer review
customer_review = """
This leaf blower is pretty amazing. It has four settings: candle blower, gentle breeze, windy city, and tornado. It arrived in two days, just in time for my wife's anniversary present. I think my wife liked it so much she was speechless. So far I've been the only one using it, and I've been using it every other morning to clear the leaves on our lawn. It's slightly more expensive than the other leaf blowers out there, but I think it's worth it for the extra features.
"""

# Template for extracting specific information
review_template = """
For the following text, extract the following information:

gift: Was the item purchased as a gift for someone else? Answer True if yes, False if not or unknown.

delivery_days: How many days did it take for the product to arrive? If this information is not found, output -1.

price_value: Extract any sentences about the value or price, and output them as a comma separated Python list.

Format the output as JSON with the following keys:
gift
delivery_days
price_value

text: {text}
"""

# Create prompt template
prompt_template = ChatPromptTemplate.from_template(review_template)
messages = prompt_template.format_messages(text=customer_review)

# Run LLM to extract structured data
response = chat(messages)
print(response.content)

# --------------------------
# Example 3: Using StructuredOutputParser
# --------------------------

# Define response schemas for structured output
gift_schema = ResponseSchema(name="gift", description="Was the item purchased as a gift for someone else? Answer True if yes, False if not or unknown.")
delivery_days_schema = ResponseSchema(name="delivery_days", description="How many days did it take for the product to arrive? If this information is not found, output -1.")
price_value_schema = ResponseSchema(name="price_value", description="Extract any sentences about the value or price, and output them as a comma separated Python list.")

# Combine schemas into a parser
response_schemas = [gift_schema, delivery_days_schema, price_value_schema]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()

# Enhanced review template with format instructions
review_template_2 = """
For the following text, extract the following information:

gift: Was the item purchased as a gift for someone else? Answer True if yes, False if not or unknown.

delivery_days: How many days did it take for the product to arrive? If this information is not found, output -1.

price_value: Extract any sentences about the value or price, and output them as a comma separated Python list.

text: {text}

{format_instructions}
"""

# Create the prompt with formatting instructions
prompt = ChatPromptTemplate.from_template(template=review_template_2)
messages = prompt.format_messages(text=customer_review, format_instructions=format_instructions)

# Display the formatted prompt
print(messages[0].content)

# Get structured LLM response
response = chat(messages)
output_dict = output_parser.parse(response.content)

# Display parsed results
print(output_dict)
print(output_dict.get('delivery_days'))
