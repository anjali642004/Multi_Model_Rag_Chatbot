from langchain_community.llms import LlamaCpp
# Simplified imports - removed complex chain dependencies

from src.utils import load_config


class LlamaChain:
    def __init__(self, chat_memory) -> None:
        # Simplified approach - no complex memory or chains
        self.memory = chat_memory
        config = load_config()
        self.llm = LlamaCpp(**config['chat_model'])

    def run(self, user_input):
        # Simple direct LLM call
        response = self.llm.invoke(user_input)
        return response