import tiktoken
import os

os.environ["TIKTOKEN_CACHE_DIR"] = r"C:\Users\Kalpit\Documents\Code\Rene 2\Test Things\RAG Testing\LangChain_RAG\assets"

# Documents
question = "What kinds of pets do I like?"
document = "My favorite pet is a cat."

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    # Returns number of tokens in a text string
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

print(num_tokens_from_string(question, "cl100k_base"))