import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

load_dotenv()


def get_llm(local_llm = True, model_name="qwen2.5:7b"):
    """
    Initialize the LLM with the specified parameters.
    """
    if local_llm:
        llm = ChatOllama(model=model_name, base_url="http://localhost:11434")
    else:
        llm = ChatOpenAI(
            model_name=model_name,
            api_key=os.getenv("OPENAI_API_KEY"),
        )
    return llm
