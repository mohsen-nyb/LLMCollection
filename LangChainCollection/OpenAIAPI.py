import os
import openai

os.environ['OPENAI_API_KEY']= "" # fill in with your openai api aky. you can get it from here: https://platform.openai.com/api-keys
openai.api_key = os.environ['OPENAI_API_KEY']
llm_model = "gpt-3.5-turbo"



# Direct API calls to OpenAI
def get_completion(prompt, model = llm_model):
  client = openai.OpenAI()
  response = client.chat.completions.create(
      model = model,
      messages = [{"role": "user", "content":prompt}],
      temperature = 0.7,
  )
  return response.choices[0].message.content

print(get_completion("What is 2+2?"))


customer_email = """
Arrr, I be fuming that me blender lid \
flew off and splattered me kitchen walls \
with smoothie! And to make matters worse,\
the warranty don't cover the cost of \
cleaning up me kitchen. I need yer help \
right now, matey!
"""

style = """American English \
in a calm and respectful tone
"""

prompt = f"""Translate the text \
that is delimited by triple backticks 
into a style that is {style}.
text: ```{customer_email}```
"""

response = get_completion(prompt)
response
