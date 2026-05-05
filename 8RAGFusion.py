import os 

os.environ['USER_AGENT'] = 'LangChain_RAG'
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = '<hey_what_you_watching? get_your_api_key!_you_vibe_coder>'

# imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.load import dumps, loads
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter

# Indexing
# Load Documents
DOC_PATH = "constitution_of_india.pdf"
loader = UnstructuredPDFLoader(file_path=DOC_PATH)

docs = loader.load()

# Spilts
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size = 300,
    chunk_overlap=20
)

# # Make Splits
splits = text_splitter.split_documents(docs)

# # Local embeddings
EMBEDDING_MODEL = "nomic-embed-text"
embeddings=OllamaEmbeddings(model=EMBEDDING_MODEL)

# Index
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

# Prompt
template = '''
You are a helpful assistant that generates multiple search queries based on a single input query. \n
Generate multiple search queries related to: {question} \n
Output (4 queries):'''
prompt_rag_fusion = ChatPromptTemplate.from_template(template)

local_model = "deepseek-r1:8b"
generate_queries = (
    prompt_rag_fusion
    | ChatOllama(model=local_model, temperature=0)
    | StrOutputParser()
    | (lambda x: x.split("\n"))
)

def reciprocal_rank_fusion(results: list[list], k=60):
    """Reciprocal_rank_fusion that takes multiple lists of ranked documents
    and an optional parameter k used in the RRF formula
    """
    
    # Initialize a dictionary to hold fused scores for each unique document
    fused_scores = {}
    
    # Iterate through each list of ranked documents
    for docs in results:
        fused_scores = {}
        # Iterate through each document in the list, with its rank (position in the list)
        for rank, doc in enumerate(docs):
            # Convert the document to a string format to use as a key (assumes documents can be serialized to JSON)
            doc_str = dumps(doc)
            # If the document is not yet in the fused_scores dictionary, add it with an initial score of 0
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            # Retrieve the current score of the document, if any
            previous_score = fused_scores[doc_str]
            #Update the score of the document using the RRF formula: 1 / (rank + k)
            fused_scores[doc_str] += 1/(rank+k)
            
    # Sort the documents based on their fused scores in descending order to get the final reranked results
    reranked_results = [
        # (loads, score)
        loads(doc_str)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]
            
    # Return the reranked results as a list of tuples, each containing the document and its fused score
    return reranked_results

question = "tell me about Panchayats."

retrieve_chain_rag_fusion = generate_queries | retriever.map() | reciprocal_rank_fusion
docs = retrieve_chain_rag_fusion.invoke({"question": question})
print(len(docs))

# RAG
template_rag = """Answer the following question based on this context:

{context}

Question: {question}
"""

prompt_rag = ChatPromptTemplate.from_template(template_rag)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

final_rag_chain = (
    {"context": retrieve_chain_rag_fusion | format_docs, # Format to string here
     "question": itemgetter("question")}
    | prompt_rag
    | ChatOllama(model=local_model)
    | StrOutputParser()
)

final_rag_chain.invoke({"question": question})