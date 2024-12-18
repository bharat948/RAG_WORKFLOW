import ast
import csv
import os
import io
import re

from logger import setup_logging
import fitz
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import torch
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, util
import time
import textwrap
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List

from embed_model import MPNetEncoder
model_dir = "embedModel"
import logging
import os
from logging.handlers import RotatingFileHandler
import traceback
logger = setup_logging()
class PDFProcessor:
    @staticmethod
    def text_formatter(text: str) -> str:
        try:
            formatted_text = text.replace("\n", " ").strip()
            logger.debug(f"Text formatted: {len(formatted_text)} characters")
            return formatted_text
        except Exception as e:
            logger.error(f"Error formatting text: {e}")
            return text

    @staticmethod
    def process_pdf(pdf_file: bytes, filename: str) -> list[dict]:
        try:
            logger.info(f"Processing PDF: {filename}")
            doc = fitz.open(stream=pdf_file, filetype="pdf")
            pages_and_texts = []
            
            for page_number, page in enumerate(doc):
                try:
                    text = page.get_text()
                    text = PDFProcessor.text_formatter(text)
                    pages_and_texts.append({
                        "filename": filename,
                        "page_number": page_number + 1,
                        "page_char_count": len(text),
                        "page_word_count": len(text.split(" ")),
                        "page_sentence_count_raw": len(text.split(". ")),
                        "page_token_count": len(text) / 4,
                        "text": text
                    })
                except Exception as page_error:
                    logger.warning(f"Error processing page {page_number + 1} in {filename}: {page_error}")
            
            logger.info(f"PDF {filename} processed successfully. Total pages: {len(pages_and_texts)}")
            return pages_and_texts
        except Exception as e:
            logger.error(f"Critical error processing PDF {filename}: {e}")
            logger.error(traceback.format_exc())
            raise

class SentenceProcessor:
    def __init__(self):
        try:
            nltk.download('punkt', quiet=True)
            logger.info("Sentence Processor initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing SentenceProcessor: {e}")
            raise

    def process_sentences(self, pages_and_texts):
        try:
            logger.info("Starting sentence processing")
            for item in tqdm(pages_and_texts, desc="Processing Sentences"):
                item["sentences"] = sent_tokenize(item["text"])
                item["sentences"] = [str(sentence) for sentence in item["sentences"]]
                item["page_sentence_count_nltk"] = len(item["sentences"])
            
            logger.info(f"Sentence processing completed. Total items processed: {len(pages_and_texts)}")
            return pages_and_texts
        except Exception as e:
            logger.error(f"Error in sentence processing: {e}")
            logger.error(traceback.format_exc())
            raise

class ChunkCreator:
    @staticmethod
    def split_list(input_list: list, slice_size: int) -> list[list[str]]:
        # print("split created")
        return [input_list[i:i + slice_size] for i in range(0, len(input_list), slice_size)]

    @staticmethod
    def create_chunks(pages_and_texts, num_sentence_chunk_size=10):
        try:
            logger.info(f"Creating chunks with {num_sentence_chunk_size} sentences per chunk")
            
            for item in tqdm(pages_and_texts, desc="Creating Sentence Chunks"):
                item["sentence_chunks"] = ChunkCreator.split_list(
                    input_list=item["sentences"],
                    slice_size=num_sentence_chunk_size
                )
                item["num_chunks"] = len(item["sentence_chunks"])
            
            pages_and_chunks = []
            for item in tqdm(pages_and_texts, desc="Processing Chunks"):
                for sentence_chunk in item["sentence_chunks"]:
                    chunk_dict = {"page_number": item["page_number"]}
                    
                    # Join sentences
                    joined_sentence_chunk = "".join(sentence_chunk).replace("  ", " ").strip()
                    joined_sentence_chunk = re.sub(r'\.([A-Z])', r'. \1', joined_sentence_chunk)
                    
                    chunk_dict["sentence_chunk"] = joined_sentence_chunk
                    chunk_dict["chunk_char_count"] = len(joined_sentence_chunk)
                    chunk_dict["chunk_word_count"] = len(joined_sentence_chunk.split(" "))
                    chunk_dict["chunk_token_count"] = len(joined_sentence_chunk) / 4
                    
                    pages_and_chunks.append(chunk_dict)
            
            logger.info(f"Chunk creation completed. Total chunks created: {len(pages_and_chunks)}")
            return pages_and_chunks
        except Exception as e:
            logger.error(f"Error creating chunks: {e}")
            logger.error(traceback.format_exc())
            raise

class EmbeddingCreator:
    def __init__(self):
        try:
            self.model_dir = "embedModel"
            self.model = MPNetEncoder(self.model_dir)
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            raise

    def create_embeddings(self, chunks):
        try:
            logger.info("Starting embedding creation")
            df = pd.DataFrame(chunks)
            
            # Filter chunks based on token length
            min_token_length = 30
            pages_and_chunks_over_min_token_len = df[df["chunk_token_count"] > min_token_length].to_dict(orient="records")
            
            logger.info(f"Total chunks over {min_token_length} tokens: {len(pages_and_chunks_over_min_token_len)}")
            
            for item in tqdm(pages_and_chunks_over_min_token_len, desc="Generating Embeddings"):
                try:
                    item["embedding"] = " ".join(map(str, self.model.encode(item["sentence_chunk"])))
                except Exception as embedding_error:
                    logger.warning(f"Could not create embedding for chunk: {embedding_error}")
            
            logger.info("Embedding creation completed")
            return pages_and_chunks_over_min_token_len
        except Exception as e:
            logger.error(f"Error in embedding creation: {e}")
            logger.error(traceback.format_exc())
            raise

class QueryProcessor:
    def __init__(self):
        self.model_dir = "embedModel"
        # self.loader = ModelLoader(self.model_dir)
        self.embedding_model = MPNetEncoder(self.model_dir)
        # self.embedding_model = embedding_model
        print("model loaded successfully") 
    def get_embedding_from_csv(self,csv_path):
        text_chunks_and_embedding_df = pd.read_csv(csv_path)
        # print(text_chunks_and_embedding_df["pages and chunk"])
        text_chunks_and_embedding_df["embedding"] = text_chunks_and_embedding_df["embedding"].fillna("[]")

        # text_chunks_and_embedding_df["embedding"] = text_chunks_and_embedding_df["embedding"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))
        text_chunks_and_embedding_df["embedding"] = text_chunks_and_embedding_df["embedding"].apply(
        lambda x: np.fromstring(x.strip("[]"), sep=" ")
        )
        # max_len = max(text_chunks_and_embedding_df["embedding"].apply(len))

        # Pad or truncate all embeddings to have the same length
        def pad_or_truncate(arr, target_dim=768):
            if len(arr) > target_dim:
                return arr[:target_dim]
            elif len(arr) < target_dim:
                return np.pad(arr, (0, target_dim - len(arr)), mode='constant')
            return arr

        text_chunks_and_embedding_df["embedding"] = text_chunks_and_embedding_df["embedding"].apply(
            lambda x: pad_or_truncate(x)
        )

        # Convert to torch tensor
        
        # Convert texts and embedding df to list of dicts
        embeddings = torch.tensor(
        np.array(text_chunks_and_embedding_df["embedding"].tolist()), 
        dtype=torch.float32
        ).to('cpu')

    # Convert texts and embedding df to list of dicts
        pages_and_chunks = text_chunks_and_embedding_df.to_dict(orient="records")
        return embeddings,pages_and_chunks

    def retrieve_relevant_resources(self,query: str,
                                embeddings: torch.tensor,
                                model: SentenceTransformer,
                                n_resources_to_return: int=5,
                                print_time: bool=True):
        """
        Embeds a query with model and returns top k scores and indices from embeddings.
        """

    # Embed the query
        query_embedding = model.encode(query)
        query_embedding = torch.tensor(query_embedding, dtype=torch.float32).view(1, -1)
        # Get dot product scores on embeddings
        if embeddings.dim() == 1:
            embeddings = embeddings.unsqueeze(0)
        start_time = time.time()
        dot_scores = util.dot_score(query_embedding, embeddings)[0]
        end_time = time.time()

        if print_time:
            print(f"[INFO] Time taken to get scores on {len(embeddings)} embeddings: {end_time-start_time:.5f} seconds.")

        scores, indices = torch.topk(input=dot_scores,
                                    k=n_resources_to_return)

        return scores, indices

    def print_wrapped(self,text, wrap_length=80):
        wrapped_text = textwrap.fill(text, wrap_length)
        print(wrapped_text)


    def prompt_formatter(self,query: str,
                        context_items: list[dict]) -> str:
        """
        Augments query with text-based context from context_items.
        """
        # Join context items into one dotted paragraph
        context = "- " + "\n- ".join([item["sentence_chunk"] for item in context_items])

        # Create a base prompt with examples to help the model
        # Note: this is very customizable, I've chosen to use 3 examples of the answer style we'd like.
        # We could also write this in a txt file and import it in if we wanted.
        base_prompt = """Based on the following context items, please answer the query.
    Give yourself room to think by extracting relevant passages from the context before answering the query.
    Don't return the thinking, only return the answer.
    Make sure your answers are as explanatory as possible.
    Use the following examples as reference for the ideal answer style.
    \nExample 1:
    Query: What are the fat-soluble vitamins?
    Answer: The fat-soluble vitamins include Vitamin A, Vitamin D, Vitamin E, and Vitamin K. These vitamins are absorbed along with fats in the diet and can be stored in the body's fatty tissue and liver for later use. Vitamin A is important for vision, immune function, and skin health. Vitamin D plays a critical role in calcium absorption and bone health. Vitamin E acts as an antioxidant, protecting cells from damage. Vitamin K is essential for blood clotting and bone metabolism.
    \nExample 2:
    Query: What are the causes of type 2 diabetes?
    Answer: Type 2 diabetes is often associated with overnutrition, particularly the overconsumption of calories leading to obesity. Factors include a diet high in refined sugars and saturated fats, which can lead to insulin resistance, a condition where the body's cells do not respond effectively to insulin. Over time, the pancreas cannot produce enough insulin to manage blood sugar levels, resulting in type 2 diabetes. Additionally, excessive caloric intake without sufficient physical activity exacerbates the risk by promoting weight gain and fat accumulation, particularly around the abdomen, further contributing to insulin resistance.
    \nExample 3:
    Query: What is the importance of hydration for physical performance?
    Answer: Hydration is crucial for physical performance because water plays key roles in maintaining blood volume, regulating body temperature, and ensuring the transport of nutrients and oxygen to cells. Adequate hydration is essential for optimal muscle function, endurance, and recovery. Dehydration can lead to decreased performance, fatigue, and increased risk of heat-related illnesses, such as heat stroke. Drinking sufficient water before, during, and after exercise helps ensure peak physical performance and recovery.
    \nNow use the following context items to answer the user query:
    {context}
    \nRelevant passages: <extract relevant passages from the context here>
    User query: {query}
    Answer:"""

        # Update base prompt with context items and query
        base_prompt = base_prompt.format(context=context, query=query)

        # Create prompt template for instruction-tuned model
        dialogue_template = [
            {"role": "user",
            "content": base_prompt}
        ]

        # Apply the chat template
        # prompt = tokenizer.apply_chat_template(conversation=dialogue_template,
        #                                       tokenize=False,
        #                                       add_generation_prompt=True)
        return base_prompt

# FastAPI app
app = FastAPI()

# Global variables
embedding_model = None
chunks_with_embeddings = []
sentence_processor = None
chunk_creator = None
embedding_creator = None
query_processor = None

class Query(BaseModel):
    text: str
    n_results: int = 5
    document_name: str

from fastapi import FastAPI, UploadFile

app = FastAPI()


@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...),document_name:str=''):
    try:
        # Read the uploaded file content
        contents = await file.read()
        print("file successfully readed")
        sentence_processor = SentenceProcessor()
        chunk_creator = ChunkCreator()
        embedding_creator = EmbeddingCreator()
        pages_and_texts = PDFProcessor.process_pdf(contents, file.filename)
        pages_and_texts = sentence_processor.process_sentences(pages_and_texts)
        # print("pages and text created")
        chunks = chunk_creator.create_chunks(pages_and_texts)
        # print("after chunks")
        pages_and_chunks_over_min_token_len = embedding_creator.create_embeddings(chunks)
        df = pd.DataFrame(pages_and_chunks_over_min_token_len)
        print(df.head())
        output_file_path = os.path.join("data", f"{document_name}_embeddings.csv")

        if os.path.exists(output_file_path):

            print(f"File '{output_file_path}' already exists. Skipping creation.")
            return pages_and_chunks_over_min_token_len
       
        df.to_csv(output_file_path, index=False, encoding='utf-8')

        print(f"Embeddings saved to {output_file_path}")
        return {"message": f"PDF {file.filename} processed successfully"}

    except Exception as e:
        # Handle potential errors during processing
        print(f"Error processing PDF: {e}")
        return {"error": f"An error occurred while processing the PDF: {str(e)}"}
    
@app.post("/upload_multiple_pdfs")
async def upload_multiple_pdfs(files: List[UploadFile] = File(...),document_name:str=''):
    global chunks_with_embeddings
    all_chunks_with_embeddings = []
    for file in files:
        contents = await file.read()
        print("file successfully readed")
        sentence_processor = SentenceProcessor()
        chunk_creator = ChunkCreator()
 
        embedding_creator = EmbeddingCreator()
        pages_and_texts = PDFProcessor.process_pdf(contents, file.filename)
        pages_and_texts = sentence_processor.process_sentences(pages_and_texts)
        # print("pages and text created")
        chunks = chunk_creator.create_chunks(pages_and_texts)
        # print("after chunks")
        pages_and_chunks_over_min_token_len = embedding_creator.create_embeddings(chunks)
        all_chunks_with_embeddings.extend(pages_and_chunks_over_min_token_len)
    if all_chunks_with_embeddings:
        df = pd.DataFrame(all_chunks_with_embeddings)
        output_file_path = os.path.join("data", f"{document_name}_cumulative_embeddings.csv")
        
        if os.path.exists(output_file_path):
            print(f"File '{output_file_path}' already exists. Appending new embeddings.")
            df.to_csv(output_file_path, mode='a', index=False, header=False, encoding='utf-8')
        else:
            df.to_csv(output_file_path, index=False, encoding='utf-8')
        
        print(f"Cumulative embeddings saved to {output_file_path}")
        return {"message": f"{len(files)} PDFs processed and embeddings saved successfully"}
    
    return {"message": f"{len(files)} PDFs processed and added to the database"}
from fastapi import HTTPException, Request



import os

from groq import Groq

# api_key = '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'
# client = Groq(
#     api_key=api_key
# )
# def response_from_groq(primer , prompt):
#     chat_completion = client.chat.completions.create(
#         messages=[
#             {
#                 "role": "system",
#                 "content": primer,
#             },
#             {
#                 "role": "user",
#                 "content": prompt
#             }
#         ],
#         model="llama3-70b-8192",
#         temperature=0.095
#     )
#     print("response generated")

#     return chat_completion.choices[0].message.content
@app.post("/query")
async def process_query(request:Request, query: Query):
    body = await request.json()
    document_name = body.get('document_name', '')
    model_dir = "embedModel"
        # self.loader = ModelLoader(self.model_dir)
    embedding_model = MPNetEncoder(model_dir)
        # self.embedding_model = embedding_model
    print("model loaded successfully") 
    if not document_name:
        raise HTTPException(status_code=400, detail="Document name is required.")
    print(document_name,query)
    query_processor = QueryProcessor()
    data_directory = "data"  # Assuming your CSV files are stored in a "data" directory
    csv_file_path = os.path.join(data_directory, f"{document_name}.csv")
    if not os.path.exists(csv_file_path):
        raise HTTPException(status_code=404, detail=f"File '{document_name}.csv' not found in the data directory.")
    try:
        embedding,pages_and_chunks = query_processor.get_embedding_from_csv(csv_path=csv_file_path)    
    
  
        if query_processor is None :
            raise HTTPException(status_code=500, detail="Server is not ready or no PDFs have been uploaded. Please try again later.")
        scores,indices=query_processor.retrieve_relevant_resources(query.text,embeddings=embedding,model=embedding_model)
        # results = query_processor.retrieve_relevant_resources(query.text, chunks_with_embeddings, query.n_results)
        context_items = [pages_and_chunks[i] for i in indices]
        for i, item in enumerate(context_items):
            item["score"] = scores[i]
        prompt = query_processor.prompt_formatter(query=query,
                              context_items=context_items)
        # response = response_from_groq(primer=prompt, prompt=query.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading file '{csv_file_path}': {str(e)}")

    return {"query": query.text, "results": prompt}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)