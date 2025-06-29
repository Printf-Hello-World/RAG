from typing import Any, List, Dict, Callable, Optional
from pydantic import BaseModel
from unstructured.partition.pdf import partition_pdf
from html_to_markdown import convert_to_markdown
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import base64

import os

class DataLoader():
    """
    This class loads in the data from a source or sources, and chooses the correct method to process it
    """
    def __init__(self, sources:List = None, text_output_path:str = None, img_output_path: str = None, model_name: str = None):
        self.sources = sources
        self.img_output_path = img_output_path
        self.text_output_path = text_output_path
        self.extracted_texts = []
        self.function_map = {
            ".pdf":self.preprocess_pdf
        }
        self.model_name = model_name
        self.img_descriptions =[]
        self.table_descriptions =[]
        self.img_prompt = "Describe the image in detail. Be specific about graphs, such as bar plots."
        self.table_prompt = """

            You are a data summarization assistant for a retrieval-based AI system.

            Analyze the following markdown table and create a concise summary that captures:
            - The subject and purpose of the table
            - Key numeric trends, comparisons, or outliers (include numbers where helpful)
            - The domain or context (e.g., sales, medical, finance, research)
            - Any insights that would help match future search queries

            Table:
            """

    def process_all(self):
        """
        processes all the different data sources, returns the doc_ids, img_ids, table_ids for vector database storage
        """
        # iterate through all the file paths
        for files in self.sources:
            # get the file extensions
            base, ext = os.path.splitext(files)
            self.function_map[ext](files, img_output_path = self.img_output_path)

        # save each chunk as a separate txt file
        for i, text in enumerate(self.extracted_texts):
            filename = os.path.join(self.text_output_path, f"chunk_{i}.txt")
            with open(filename, "w", encoding="utf-8") as f:
                f.write(text)

        
    # use the unstructed library to handle pdfs, 
    def preprocess_pdf(self, source, **kwargs):
        """
        This function handles the pdf preprocessing, uses unstructured library
        if there are images, need to specify the output path

        returns the extracted texts from tables or texts
        """
        # Get elements
        img_output_path = Path(kwargs.get("img_output_path", None))
        img_output_path.mkdir(parents=True, exist_ok=True)
        

        raw_pdf_elements = partition_pdf(
            filename=source,
            # Using pdf format to find embedded image blocks
            extract_images_in_pdf=True,
            # Use layout model (YOLOX) to get bounding boxes (for tables) and find titles
            # Titles are any sub-section of the document
            infer_table_structure=True,
            # Post processing to aggregate text once we have the title
            chunking_strategy="by_title",
            strategy="hi_res",
            extract_image_block_types=["Image", "Table"],
            # Chunking params to aggregate text blocks
            # Attempt to create a new chunk 3800 chars
            # Attempt to keep chunks > 2000 chars
            # Hard max on chunks
            max_characters=4000,
            new_after_n_chars=3800,
            combine_text_under_n_chars=2000,
            extract_image_block_to_payload=False,   
            extract_image_block_output_dir=img_output_path,
        )
      
        # if there are tables, skip them because we already saved them as images
        for element in raw_pdf_elements:
            if "unstructured.documents.elements.Table" in str(type(element)):
                continue
            else:
                self.extracted_texts.append(str(element))

        # initialise the model for image and table processing
        model = ChatOpenAI(model = self.model_name)

        # need to process the images and tables, get their descriptions
        for file_path in Path(img_output_path).iterdir():
            if file_path.is_file():
                if file_path.name.startswith("table"):
                    # get llm output
                    des = self.get_llm_descriptions(file_path, model, self.table_prompt)
                    # append output to the list
                    self.table_descriptions.append(des)
                    # save output to the correct file name
                    self.save_descriptions(file_path, Path(img_output_path), des)
                # repeat for figures
                elif file_path.name.startswith("figure"):
                    des = self.get_llm_descriptions(file_path, model, self.img_prompt)
                    self.img_descriptions.append(des)
                    self.save_descriptions(file_path,Path(img_output_path), des)

    # uses an LLm to get a summary of a table or figure
    def get_llm_descriptions(self,img_path, model, prompt):
        with open(img_path, "rb") as image_file:
            image_data = self.encode_image(img_path)

        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                },
            ],
        )

        response = model.invoke([message])
        return response.content
    
    # helper function to encode images
    def encode_image(self, image_path):
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    # helper function to save llm outputs of images and table summaries
    def save_descriptions(self, file_path,file_directory, descriptions):
        # Get the base name, e.g., "image1.jpg"
        base_name = os.path.basename(file_path)
        name_without_ext = os.path.splitext(base_name)[0]
        txt_filename = f"{name_without_ext}.txt"
        
        # Full path to where you want to save the .txt file
        txt_file_path = os.path.join(file_directory, txt_filename)


        with open(txt_file_path, "w") as text_file:
            text_file.write(descriptions)

    



        