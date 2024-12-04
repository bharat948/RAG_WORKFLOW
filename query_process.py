import textwrap
from embed_model import MPNetEncoder
import numpy as np
import torch

import pandas as pd
from sentence_transformers import SentenceTransformer, util
import time
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

        text_chunks_and_embedding_df["embedding"] = text_chunks_and_embedding_df["embedding"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))
        max_len = max(text_chunks_and_embedding_df["embedding"].apply(len))

        # Pad or truncate all embeddings to have the same length
        def pad_or_truncate(arr, max_len):
            if len(arr) > max_len:
                return arr[:max_len]
            elif len(arr) < max_len:
                return np.pad(arr, (0, max_len - len(arr)), mode='constant')
            return arr

        text_chunks_and_embedding_df["embedding"] = text_chunks_and_embedding_df["embedding"].apply(
            lambda x: pad_or_truncate(x, max_len)
        )

        # Convert to torch tensor
        
        # Convert texts and embedding df to list of dicts
        pages_and_chunks = text_chunks_and_embedding_df.to_dict(orient="records")

        # Convert embeddings to torch tensor and send to device (note: NumPy arrays are float64, torch tensors are float32 by default)
        embeddings = torch.tensor(np.array(text_chunks_and_embedding_df["embedding"].tolist()), dtype=torch.float32).to('cpu')
        embeddings.shape
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

        # Get dot product scores on embeddings
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