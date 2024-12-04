import pandas as pd
from logger import setup_logging
from embed_model import MPNetEncoder
import tqdm
logger = setup_logging()
import traceback
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