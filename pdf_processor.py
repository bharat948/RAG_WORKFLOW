import fitz
import nltk
import tqdm

from nltk.tokenize import sent_tokenize
import pandas as pd
import re
class PDFProcessor:
    @staticmethod
    def text_formatter(text: str) -> str:
        return text.replace("\n", " ").strip()

    @staticmethod
    def process_pdf(pdf_file: bytes, filename: str) -> list[dict]:
        doc = fitz.open(stream=pdf_file, filetype="pdf")
        pages_and_texts = []
        for page_number, page in enumerate(doc):
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

        
        return pages_and_texts
    

class SentenceProcessor:
    def __init__(self):
        # from spacy.lang.en import English
        # self.nlp = English()
        # self.nlp.add_pipe("sentencizer")
        
        
        nltk.download('punkt_tab')
        # self.sentencer = sent_tokenize()
        print("Sentence Processor class initialized")

    def process_sentences(self, pages_and_texts):
        print("inside process sentences")
        for item in tqdm(pages_and_texts):
            # item["sentences"] = [str(sentence) for sentence in self.nlp(item["text"]).sents]
            # item["page_sentence_count_spacy"] = len(item["sentences"])
            item["sentences"] = sent_tokenize(item["text"])

            item["sentences"] = [str(sentence) for sentence in item["sentences"]]
    
    # Count the sentences
            item["page_sentence_count_nltk"] = len(item["sentences"])

        df = pd.DataFrame(pages_and_texts)
        print(df)
        return pages_and_texts



class ChunkCreator:
    @staticmethod
    def split_list(input_list: list, slice_size: int) -> list[list[str]]:
        # print("split created")
        return [input_list[i:i + slice_size] for i in range(0, len(input_list), slice_size)]

    @staticmethod
    def create_chunks(pages_and_texts, num_sentence_chunk_size=10):
        for item in tqdm(pages_and_texts):
            item["sentence_chunks"] = ChunkCreator.split_list(input_list=item["sentences"],
                                                slice_size=num_sentence_chunk_size)
            item["num_chunks"] = len(item["sentence_chunks"])
        pages_and_chunks = []
        for item in tqdm(pages_and_texts):
            for sentence_chunk in item["sentence_chunks"]:
                chunk_dict = {}
                chunk_dict["page_number"] = item["page_number"]

                # Join the sentences together into a paragraph-like structure, aka a chunk (so they are a single string)
                joined_sentence_chunk = "".join(sentence_chunk).replace("  ", " ").strip()
                joined_sentence_chunk = re.sub(r'\.([A-Z])', r'. \1', joined_sentence_chunk) # ".A" -> ". A" for any full-stop/capital letter combo
                chunk_dict["sentence_chunk"] = joined_sentence_chunk

                # Get stats about the chunk
                chunk_dict["chunk_char_count"] = len(joined_sentence_chunk)
                chunk_dict["chunk_word_count"] = len([word for word in joined_sentence_chunk.split(" ")])
                chunk_dict["chunk_token_count"] = len(joined_sentence_chunk) / 4 # 1 token = ~4 characters

                pages_and_chunks.append(chunk_dict)
        return pages_and_chunks