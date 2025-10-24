from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from src.utils import load_config
from src.vectorstore import VectorDB


def format_docs(docs: list[Document]):
    return '\n\n'.join(doc.page_content for doc in docs)


class OllamaChain:
    def __init__(self, chat_memory) -> None:
        # Simplified approach - no complex chains
        self.memory = chat_memory
        config = load_config()
        self.llm = Ollama(**config['chat_model'])

    def run(self, user_input):
        # Simple direct LLM call
        response = self.llm.invoke(user_input)
        return response


class OllamaRAGChain:
    def __init__(self, chat_memory, uploaded_file=None):
        # Simplified RAG approach
        self.vector_db = VectorDB('chroma', 'pdf_documents')
        if uploaded_file:
            self.update_knowledge_base(uploaded_file)

        config = load_config()
        self.llm = Ollama(**config['chat_model'])
        self.chat_memory = chat_memory

    def run(self, user_input):
        # Simple RAG implementation
        try:
            # Get relevant documents
            docs = self.vector_db.as_retriever().get_relevant_documents(user_input)
            context = format_docs(docs)
            
            # Create prompt with context
            prompt = f"""Context: {context}
            
Question: {user_input}

Answer based on the context above:"""
            
            response = self.llm.invoke(prompt)
            return response
        except Exception as e:
            # Fallback to simple LLM call
            return self.llm.invoke(user_input)

    def update_chain(self, uploaded_pdf):
        self.update_knowledge_base(uploaded_pdf)

    def update_knowledge_base(self, uploaded_pdf):
        self.vector_db.index(uploaded_pdf)