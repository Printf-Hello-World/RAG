from fastapi import FastAPI, UploadFile, File
import os
from typing import List
from dataloader import DataLoader
from embedder import Embedder
from vectorstore import Vectorstore
from retriever import Retriever
from generator import Generator
import os
from dotenv import load_dotenv
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import base64
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import chromadb
from pydantic import BaseModel
from .utils.config import config

load_dotenv()
embedding_model = AzureOpenAIEmbeddings(
    model=os.getenv('AZURE_OPENAI_EMBEDDING_DEPLOYMENT'),
    azure_endpoint=os.getenv('AZURE_OPENAI_EMBEDDING_ENDPOINT'),
    api_version=os.getenv('AZURE_OPENAI_EMBEDDING_API_VERSION'),
    api_key=os.getenv('AZURE_OPENAI_EMBEDDING_API_KEY')
)

llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

img_path = config.get("img_path")
text_path = config.get("text_path")
collection_name = config.get("collection_name")
db_directory = config.get("db_directory")
SAVE_DIR = config.get("uploaded_dir")
corpus_dir = config.get("corpus_dir")
vision_model_name = config.get("model_name")



app = FastAPI()

# save the uploaded docs to file
@app.post("/upload")
async def upload(files: List[UploadFile] = File(...)):
    # save the files in the save temporary save dir for processing
    os.makedirs(SAVE_DIR, exist_ok=True)
    for file in files:
        file_location = os.path.join(SAVE_DIR, file.filename)
        with open(file_location, "wb") as f:
            content = await file.read()
            f.write(content)

    # run the chain to update the directory
    # get all the filenames of uploaded docs
    upload_doc_names = []
    for file in os.listdir(SAVE_DIR):
        upload_doc_names.append(os.path.join(SAVE_DIR, file))
    # initialise dataloader 
    os.makedirs(text_path, exist_ok=True)
    os.makedirs(img_path, exist_ok=True)
    dataloader = DataLoader(upload_doc_names,text_output_path=text_path ,img_output_path=img_path, model_name=vision_model_name)
    dataloader.process_all()

    # initialise the embedder
    embedder = Embedder(embedding_model,img_path,text_path)
    # get the embeddings
    data = embedder.get_embeddings()

    # update the vectorstore
    os.makedirs(db_directory, exist_ok=True)
    vectorstore = Vectorstore(db_directory, collection_name, data)
    vectorstore.add_documents()

class QueryRequest(BaseModel):
    query: str


# initialise the retriever
retriever = Retriever(db_directory, collection_name, embedding_model)

# initialise the generator
generator = Generator(llm)

# get the prompts from the front end and return the prediction
@app.post("/generate")
async def upload(request: QueryRequest):
    results = retriever.retrieve(request.query)
    response = generator.generate(request.query, results)
    return {"response":response}

# to start, uvicorn backend:app --reload