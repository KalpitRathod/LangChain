import os 

os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = '<hey_what_you_watching? get_your_api_key!_you_vibe_coder>'

import bs4
import ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import UnstructuredPDFLoader

# Indexing

# Load Documents
DOC_PATH = "constitution_of_india.pdf"
loader = UnstructuredPDFLoader(file_path=DOC_PATH)

docs = loader.load()

# Split
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# Embed
EMBEDDING_MODEL = "nomic-embed-text"
vectorstore = Chroma.from_documents(documents=splits,embedding=OllamaEmbeddings(model=EMBEDDING_MODEL))

retriever = vectorstore.as_retriever()

# Retrieval and Generation

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

# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Question
res = rag_chain.invoke("tell me about Elections to the Panchayats?")
print(res)