import os 

os.environ['USER_AGENT'] = 'MyRAGApp/1.0'
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
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=300, chunk_overlap=50)
splits = text_splitter.split_documents(docs)

# Embed
EMBEDDING_MODEL = "nomic-embed-text"
vectorstore = Chroma.from_documents(documents=splits,embedding=OllamaEmbeddings(model=EMBEDDING_MODEL))

retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

docs = retriever.invoke("tell me about Elections to the Panchayats?")
print(len(docs))
# see the document received in LangSmith Tracing