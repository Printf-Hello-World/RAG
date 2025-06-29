from typing import Any, List, Dict, Callable, Optional
from pydantic import BaseModel
from pathlib import Path
import os
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings
import chromadb
from langchain_chroma import Chroma

import os

class Retriever():
    """
    handles the retrieving using an embedding model
    """
    def __init__(self, db_directory: str, collection_name:str, embedder:AzureOpenAIEmbeddings = None):
        self.embedder = embedder
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=embedder,
            persist_directory=db_directory  # This must match your saved location
        )
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})

    def retrieve(self, query):
        return self.retriever.invoke(query)

    