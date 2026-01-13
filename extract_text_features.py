import torch
import pandas as pd
import ast
import numpy as np
from transformers import AutoTokenizer, AutoModel
import tqdm
import os
import pickle

# Configuration
MODEL_PATH = '/data/lzy/wel/LLM_cache/models--BAAI--bge-large-en-v1.5/snapshots/d4aa6901d3a41ba39fb536a557fa166f842b0e09'
DATA_PATH = '/data/lzy/SASRec_Project/test2.csv'
OUTPUT_PATH = '/data/lzy/SASRec_Project/item_embeddings.pkl'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def extract_embeddings():
    print(f"Loading model from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModel.from_pretrained(MODEL_PATH).to(DEVICE)
    model.eval()
    
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    
    # Extract unique items and their titles
    item_to_title = {}
    
    print("Parsing items...")
    for _, row in tqdm.tqdm(df.iterrows(), total=len(df)):
        try:
            item_ids = ast.literal_eval(row['history_item_id'])
            item_titles = ast.literal_eval(row['history_item_title'])
            
            if len(item_ids) != len(item_titles):
                continue
                
            for iid, title in zip(item_ids, item_titles):
                if iid not in item_to_title:
                    item_to_title[iid] = title
        except:
            continue
            
    print(f"Found {len(item_to_title)} unique items.")
    
    # Sort by item ID to create a matrix
    max_item_id = max(item_to_title.keys())
    print(f"Max item ID: {max_item_id}")
    
    # Initialize embedding matrix
    # We use 0 for padding, so size is max_item_id + 1
    # BGE-Large-EN-v1.5 output dimension is 1024
    embedding_dim = 1024
    embedding_matrix = torch.zeros((max_item_id + 1, embedding_dim))
    
    # Batch processing
    batch_size = 32
    items = list(item_to_title.items())
    
    print("Generating embeddings...")
    with torch.no_grad():
        for i in tqdm.tqdm(range(0, len(items), batch_size)):
            batch_items = items[i:i+batch_size]
            batch_ids = [item[0] for item in batch_items]
            batch_titles = [item[1] for item in batch_items]
            
            # Tokenize
            encoded_input = tokenizer(batch_titles, padding=True, truncation=True, return_tensors='pt').to(DEVICE)
            
            # Model forward
            model_output = model(**encoded_input)
            
            # Perform pooling. In this case, cls pooling.
            sentence_embeddings = model_output[0][:, 0]
            
            # Normalize embeddings
            sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
            
            # Store in matrix
            for j, iid in enumerate(batch_ids):
                embedding_matrix[iid] = sentence_embeddings[j].cpu()
                
    # Save embeddings
    print(f"Saving embeddings to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, 'wb') as f:
        pickle.dump(embedding_matrix, f)
        
    print("Done.")

if __name__ == "__main__":
    extract_embeddings()
