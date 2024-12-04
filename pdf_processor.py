from logger import setup_logging
import fitz
import nltk
import tqdm

from nltk.tokenize import sent_tokenize
import pandas as pd
import re
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