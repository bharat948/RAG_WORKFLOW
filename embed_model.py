import os
from sentence_transformers import SentenceTransformer

# from sentence_transformers import SentenceTransformer

class MPNetEncoder:
    def __init__(self, model_path = None,model_name ='all-mpnet-base-v2'):
        """
        Initialize the encoder with a local model path.
        
        Args:
        model_path (str): The path to the directory containing the model files.
        """
        if model_path:
            self.model = SentenceTransformer(model_path)
            print('locally')
        else:
            self.model = SentenceTransformer(model_name)
            print("downloaded")
    
    def encode(self, texts, batch_size=32,convert_to_tensor=True):
        """
        Encodes the input texts into embeddings.
        
        Args:
        texts (str or list of str): The text(s) to encode.
        batch_size (int): The batch size for encoding.
        
        Returns:
        numpy.ndarray: The embeddings for the input texts.
        """
        return self.model.encode(texts, batch_size=batch_size,convert_to_tensor=convert_to_tensor)
