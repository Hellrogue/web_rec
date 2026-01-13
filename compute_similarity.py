import torch
import pickle
import numpy as np
import tqdm
import os

# Configuration
EMB_PATH = '/data/lzy/SASRec_Project/item_embeddings.pkl'
OUTPUT_PATH = '/data/lzy/SASRec_Project/item_similarity.pkl'
TOP_K = 10
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def compute_similarity():
    print(f"Loading embeddings from {EMB_PATH}...")
    with open(EMB_PATH, 'rb') as f:
        embeddings = pickle.load(f)
        
    # Ensure embeddings are on device
    if not isinstance(embeddings, torch.Tensor):
        embeddings = torch.tensor(embeddings)
    
    embeddings = embeddings.to(DEVICE)
    
    # Normalize embeddings (they should be already, but just in case)
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    
    num_items = embeddings.size(0)
    print(f"Computing similarity for {num_items} items...")
    
    # We can't compute the full matrix at once if it's too big, but 17k x 17k is fine on GPU.
    # 17408^2 * 4 bytes = ~1.2 GB.
    
    # Compute similarity matrix: [num_items, num_items]
    # sim[i, j] = dot(emb[i], emb[j])
    
    # To save memory, we can do it in chunks if needed, but let's try full batch first.
    try:
        sim_matrix = torch.matmul(embeddings, embeddings.t())
    except RuntimeError:
        print("OOM, switching to CPU...")
        embeddings = embeddings.cpu()
        sim_matrix = torch.matmul(embeddings, embeddings.t())
        
    # For each item, get top-k similar items
    # We want to exclude the item itself (similarity 1.0)
    # So we get top-(k+1) and remove the first one (itself)
    
    print("Extracting top-k similar items...")
    top_k_indices = torch.topk(sim_matrix, k=TOP_K + 1, dim=1).indices
    
    # Convert to dictionary: item_id -> list of similar item_ids
    similarity_dict = {}
    
    top_k_indices = top_k_indices.cpu().numpy()
    
    for i in tqdm.tqdm(range(num_items)):
        # The first item is usually itself (index i), but let's be safe and filter
        sim_items = top_k_indices[i]
        sim_items = [x for x in sim_items if x != i]
        
        # Take top K
        similarity_dict[i] = sim_items[:TOP_K]
        
    print(f"Saving similarity dictionary to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, 'wb') as f:
        pickle.dump(similarity_dict, f)
        
    print("Done.")

if __name__ == "__main__":
    compute_similarity()
