from typing import Any, List, Dict, Callable, Optional
from pydantic import BaseModel
from pathlib import Path
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI

import os

class Generator():
    """
    handles the retrieving using an embedding model
    """

    def __init__(self, model: AzureChatOpenAI):
        self.model = model

    def generate(self, query: str, retrieved_docs: List):
        retrieved_docs_txt = [i.page_content for i in retrieved_docs]
        prompt = self.prepare_prompt(query, retrieved_docs_txt)
        response = self.model.invoke(prompt)
        return response.content
    


    def prepare_prompt(self, query, retrieved_docs):
        system_prompt = (
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )
        
        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )

        return prompt_template.invoke({"context": retrieved_docs, "input": query})
