# Pinecone removed - using Chroma only for local processing
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.indexes import SQLRecordManager, index

from src.pdf_handler import extract_pdf, load_pdf_directory, split_pdf

import os
import shutil
from dotenv import load_dotenv

load_dotenv()


# Pinecone setup function removed - using Chroma only


def setup_chroma(index_name, embedding_model, persist_directory=None):
    if not persist_directory:
        persist_directory = './.cache/database'

    os.makedirs(persist_directory, exist_ok=True)

    db = Chroma(index_name, embedding_function=embedding_model, persist_directory=persist_directory)
    return db


class VectorDB:
    def __init__(self, db_name, index_name, cache_dir=None):
        embedding = OllamaEmbeddings(model='nomic-embed-text:latest', num_gpu=1)

        if not cache_dir:
            cache_dir = './.cache/database'
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        # Using Chroma only for local processing
        self.vectorstore = setup_chroma(index_name, embedding, self.cache_dir)

        namespace = f'{db_name}/{index_name}'
        self.record_manager = SQLRecordManager(namespace,
                                               db_url=f'sqlite:///{self.cache_dir}/record_manager_cache.sql')
        self.record_manager.create_schema()

    def index(self, uploaded_file):
        directory = extract_pdf(uploaded_file)
        docs = load_pdf_directory(directory)
        chunks = split_pdf(docs)

        index(
            docs_source=chunks,
            record_manager=self.record_manager,
            vector_store=self.vectorstore,
            cleanup='full',
            source_id_key='source'
        )

        for file in os.listdir(directory):
            os.remove(os.path.join(directory, file))

    def as_retriever(self):
        return self.vectorstore.as_retriever()

    def __del__(self):
        shutil.rmtree(self.cache_dir)
