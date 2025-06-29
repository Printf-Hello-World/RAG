from typing import Any, List, Dict, Callable, Optional
from pydantic import BaseModel
from pathlib import Path
import os
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings


import os

class Embedder():
    """
    This class handles the embedding of the text data before feeding them to the vector store
    """
    def __init__(self, embedder:AzureOpenAIEmbeddings, img_path:str, txt_path:str):
        self.img_path = img_path
        self.txt_path = txt_path
        self.embedder = embedder
        self.texts = []
        self.embeddings = []
        self.sources =[]

    def get_embeddings(self):
        """
        this function retrieves the scrapped text and img data, embeds them and saves them, returns a tuple
        """
        # get all image and table data
        for file in Path(self.img_path).glob("*.txt"):
            with open(file, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                self.embed(content, file.name)
        # get the text data
        for file in Path(self.txt_path).glob("*.txt"):
            with open(file, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                self.embed(content, file.name)

        return (self.texts, self.embeddings, self.sources)


    def embed(self, content, file_name):
        self.texts.append(content)
        self.embeddings.append(self.embedder.embed_query(content))
        self.sources.append(file_name)


