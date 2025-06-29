from typing import Any, List, Dict, Callable, Optional
from pydantic import BaseModel
from pathlib import Path
import os
from dotenv import load_dotenv
import chromadb
from langchain_openai import AzureOpenAIEmbeddings
from chromadb.config import Settings
import uuid


class Vectorstore():
    """
    This class connects to the vector store, can add documents to it
    """
    def __init__(self, db_directory: str, collection_name:str, data):
        self.db_directory = db_directory
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(db_directory)
        self.collection = self.client.get_or_create_collection(name=collection_name)
        self.data = data

    def add_documents(self):
        self.collection.add(
            documents=self.data[0],
            metadatas=[{"source": i} for i in self.data[2]],
            embeddings=self.data[1],
            ids=[str(uuid.uuid4()) for _ in self.data[0]]
        )
        print(f"Total docs in collection: {self.collection.count()}")

