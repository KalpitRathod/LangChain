import os 

os.environ['USER_AGENT'] = 'LangChain_RAG'
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = '<hey_what_you_watching? get_your_api_key!_you_vibe_coder>'

'''
    Sementic search on the embeddings is hard to get. Embedding long documents is a challenge.
    User queries are a challenge, if a user provides an ambiguous query they'll get ambiguous matchs
    
    LLMs just follow what was in the context and hallucinate answers as a result
    
    as per "6rag_basics2.py"
    
    Few approach is Query Rewriting or RAG-Fusion.
'''

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
from langchain_core.load import dumps, loads
from langchain_core.documents import Document
from operator import itemgetter
import re

# Indexing
# Load Documents
DOC_PATH = "ArtOfLiving.pdf"
loader = UnstructuredPDFLoader(file_path=DOC_PATH)

docs = loader.load()

# Spilts
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size = 300,
    chunk_overlap=20
)

# # Make Splits
splits = text_splitter.split_documents(docs)
print(splits[0])

# # Local embeddings
EMBEDDING_MODEL = "nomic-embed-text"
embeddings=OllamaEmbeddings(model=EMBEDDING_MODEL)

# Index
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

template = """You are an AI language model assistant. Your task is to generate five different 
versions of the given user question to retrieve relevant documents from a vector database. 
Provide these alternative questions only, separated by newlines. 
Do not include any introductory text, numbers, or conclusions.

Original question: {question}"""

prompt = ChatPromptTemplate.from_template(template)

def clean_queries(queries):
    # Remove empty strings and strip leading numbers/whitespace
    cleaned = [re.sub(r"^\d+\.\s*", "", q).strip() for q in queries if q.strip()]
    return cleaned

def print_queries(queries):
    print("\n--- Generated Queries ---")
    for i, q in enumerate(queries):
        print(f"{i+1}: {q}")
    print("-------------------------\n")
    return queries

local_model = "deepseek-r1:8b"
generate_queries = (
    prompt
    | ChatOllama(model=local_model)
    | StrOutputParser()
    | (lambda x: x.split("\n"))
    | clean_queries
    | print_queries
)

'''
def get_unique_union(documents: list[list[Document]]) -> list[Document]:
documents = [
    [Document(...), Document(...)],
    [Document(...)]
]
'''
def get_unique_union(documents):
    """
    Takes a list of lists of Documents, flattens them, 
    and removes duplicates based on content/metadata.
    """
    # Flatten and serialize to JSON strings for hashing
    flattened_docs = []
    for sublist in documents:
        for doc in sublist:
            flattened_docs.append(dumps(doc))
    
    # Use set to find unique strings
    unique_docs = list(set(flattened_docs))
    
    # Deserialize back to Document objects
    result = []
    for doc in unique_docs:
        result.append(loads(doc))
    return result

# .map() tells LangChain to run the retriever for EVERY string in the input list
retrieval_chain = generate_queries | retriever.map() | get_unique_union

question = "What Makes Us Truly Happy?"
unique_docs = retrieval_chain.invoke({"question": question})

print(f"Retrieved {len(unique_docs)} unique documents.")

# Prompt
template = '''
You are an assistant. Use the following pieces of retrieved context. and respond to the question.
Question: {question}
Context: {context}
Answer:
'''

# LLM
local_model = "deepseek-r1:8b"
llm = ChatOllama(model=local_model)

# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

prompt = ChatPromptTemplate.from_template(template)
final_rag_chain = (
    {"context": retrieval_chain | format_docs,
     "question": itemgetter("question")}
    | prompt
    | llm
    | StrOutputParser()
)

print(final_rag_chain.invoke({"question":question}))