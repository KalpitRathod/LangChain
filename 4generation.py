import os 

os.environ['USER_AGENT'] = 'LangChain_RAG'
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = '<hey_what_you_watching? get_your_api_key!_you_vibe_coder>'

# Create a vector store with a sample text
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate

EMBEDDING_MODEL = "nomic-embed-text"
embedding=OllamaEmbeddings(model=EMBEDDING_MODEL)

text = "LangChain is the framework for building context-aware reasoning applications"

# Prompt
template = '''
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question}
Context: {context}
Answer:
'''
prompt = ChatPromptTemplate.from_template(template)

local_model = "qwen3.5:4b"
llm = ChatOllama(model=local_model)

# Run
chain = prompt  | llm

chain.invoke({"context": text, "question": "What is task LangChain?"})
# see the tracing LangSmith