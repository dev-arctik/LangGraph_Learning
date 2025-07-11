# Configuration settings for the project
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from config.secret_keys import OPENAI_API_KEY

# Model Configuration
MODEL_TEMPERATURE = 0.7
MODEL_NAME = "gpt-4.1-mini-2025-04-14"
EMBEDDING_MODEL_NAME = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536  # Dimensions for the embedding model - text-embedding-3-small

# llm configuration function
def get_llm(temperature: float = MODEL_TEMPERATURE, model_name: str = MODEL_NAME):
    return ChatOpenAI(
        model=model_name, 
        openai_api_key=OPENAI_API_KEY, 
        temperature=temperature,
        streaming=True
    )

# Embeddings configuration function
def get_embeddings(embedding_model_name: str = EMBEDDING_MODEL_NAME):
    return OpenAIEmbeddings(
        model=embedding_model_name,
        api_key=OPENAI_API_KEY
    )