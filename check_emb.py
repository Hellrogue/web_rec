import pickle
import torch

path = '/data/lzy/SASRec_Project/item_embeddings.pkl'
with open(path, 'rb') as f:
    emb = pickle.load(f)
    
print(f"Shape: {emb.shape}")
print(f"Type: {type(emb)}")
print(f"First 5 rows:\n{emb[:5]}")
