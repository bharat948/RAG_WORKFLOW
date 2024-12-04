import pandas as pd
from embed_model import MPNetEncoder
import tqdm

class EmbeddingCreator:
    def __init__(self):
        self.model_dir = "embedModel"
        
        self.model = MPNetEncoder(self.model_dir)
        print("model loaded successfully") 

    def create_embeddings(self, chunks):
        print("before text chunks creation")
        df = pd.DataFrame(chunks)
        min_token_length = 30
        pages_and_chunks_over_min_token_len = df[df["chunk_token_count"] > min_token_length].to_dict(orient="records")
        print("after text chunks creation")
        for item in tqdm(pages_and_chunks_over_min_token_len):
            item["embedding"] = self.model.encode(item["sentence_chunk"])
        print("embedding created")
        df = pd.DataFrame(pages_and_chunks_over_min_token_len)
        print(df.head())
        
        return pages_and_chunks_over_min_token_len