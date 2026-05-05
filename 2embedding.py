# https://docs.langchain.com/oss/python/integrations/embeddings/ollama

from langchain_ollama import OllamaEmbeddings

# Documents
question = "What kinds of pets do I like?"
document = "My favorite pet is a cat."

EMBEDDING_MODEL = "nomic-embed-text"
embedding=OllamaEmbeddings(model=EMBEDDING_MODEL)

query_result = embedding.embed_query(question)
query_result = embedding.embed_query(document)

print(len(query_result))
