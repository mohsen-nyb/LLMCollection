#adapted from "LangChain for LLM Application Development" from "DeepLearning.AI".

#!pip install -U langchain-community
#!pip install docarray
#!pip install tiktoken

import os
import openai
import warnings

from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.embeddings import OpenAIEmbeddings
from langchain.evaluation.qa import QAGenerateChain, QAEvalChain

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set your OpenAI API key
os.environ['OPENAI_API_KEY'] = ""  # Fill in with your OpenAI API key
openai.api_key = os.environ['OPENAI_API_KEY']
llm_model = "gpt-3.5-turbo"

# Load CSV data into LangChain
file = 'OutdoorClothingCatalog_1000.csv'
loader = CSVLoader(file_path=file)
data = loader.load()

# Create embeddings and vector index for document retrieval
embedding = OpenAIEmbeddings()
index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch,
    embedding=embedding
).from_loaders([loader])

# Initialize ChatOpenAI model with zero temperature for deterministic results
llm = ChatOpenAI(temperature=0.0, model=llm_model)

# Set up RetrievalQA chain for question answering over the indexed data
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=index.vectorstore.as_retriever(),
    verbose=True,
    chain_type_kwargs={
        "document_separator": "<<<<>>>>>"
    }
)

# Define example Q&A pairs for evaluation
examples = [
    {
        "query": "Do the Cozy Comfort Pullover Set have side pockets?",
        "answer": "Yes"
    },
    {
        "query": "What collection is the Ultra-Lofty 850 Stretch Down Hooded Jacket from?",
        "answer": "The DownTek collection"
    }
]

# Generate new Q&A examples using LLM
example_gen_chain = QAGenerateChain.from_llm(ChatOpenAI(model=llm_model))
new_examples = example_gen_chain.apply_and_parse(
    [{"doc": t} for t in data[:5]]  # Generate from the first 5 documents
)

# Append generated examples to the list
for ex in new_examples:
    examples.append(ex['qa_pairs'])

# Display a generated example
print(new_examples[0])

# Run QA chain on an example query
qa.run(examples[0]["query"])

# Apply the QA chain to all examples for evaluation
predictions = qa.apply(examples)

# Evaluate the QA chain responses using LLM
eval_chain = QAEvalChain.from_llm(llm)
graded_outputs = eval_chain.evaluate(examples, predictions)

# Display evaluation results
for i, eg in enumerate(examples):
    print(f"Example {i}:")
    print("Question: " + predictions[i]['query'])
    print("Real Answer: " + predictions[i]['answer'])
    print("Predicted Answer: " + predictions[i]['result'])
    print("Predicted Grade: " + graded_outputs[i]['results'])
    print()

# Tip: Turn on debug mode with langchain.debug = True for more detailed logs
