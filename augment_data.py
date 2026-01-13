import pandas as pd
import numpy as np
import ast
import pickle
import random
import tqdm
import os

# Configuration
INPUT_FILE = '/data/lzy/SASRec_Project/test2.csv'
OUTPUT_FILE = '/data/lzy/SASRec_Project/train_augmented.csv'
SIMILARITY_PATH = '/data/lzy/SASRec_Project/item_similarity.pkl'
AUGMENT_FACTOR = 5 # Generate 5 variants per sequence

def load_similarity(path):
    print(f"Loading similarity dictionary from {path}...")
    with open(path, 'rb') as f:
        return pickle.load(f)

def semantic_replace(seq, sim_dict, prob=0.3):
    new_seq = []
    for item in seq:
        if item in sim_dict and random.random() < prob:
            # Replace with one of the top-10 similar items
            candidates = sim_dict[item]
            if candidates:
                new_seq.append(random.choice(candidates))
            else:
                new_seq.append(item)
        else:
            new_seq.append(item)
    return new_seq

def crop_sequence(seq, prob=0.5):
    if len(seq) < 5:
        return seq
    if random.random() < prob:
        # Crop from beginning
        crop_len = random.randint(1, int(len(seq) * 0.4))
        return seq[crop_len:]
    return seq

def reorder_sequence(seq, prob=0.5):
    if len(seq) < 5:
        return seq
    if random.random() < prob:
        # Pick a sub-window to shuffle
        window_size = min(5, len(seq) // 2)
        start = random.randint(0, len(seq) - window_size)
        sub = seq[start : start + window_size]
        random.shuffle(sub)
        new_seq = list(seq)
        new_seq[start : start + window_size] = sub
        return new_seq
    return seq

def insert_noise(seq, vocab_size, prob=0.1):
    # Randomly insert items? No, that might break semantics too much.
    # Maybe insert similar items?
    return seq

def augment_data():
    print(f"Loading data from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    
    sim_dict = None
    if os.path.exists(SIMILARITY_PATH):
        sim_dict = load_similarity(SIMILARITY_PATH)
    else:
        print("Warning: Similarity dictionary not found. Semantic augmentation will be skipped.")
    
    augmented_rows = []
    
    print(f"Augmenting data (Factor: {AUGMENT_FACTOR})...")
    for _, row in tqdm.tqdm(df.iterrows(), total=len(df)):
        user_id = row['user_id']
        try:
            seq = ast.literal_eval(row['history_item_id'])
        except:
            continue
            
        # 1. Original Sequence
        augmented_rows.append({
            'user_id': user_id,
            'history_item_id': str(seq),
            'type': 'original'
        })
        
        # 2. Augmented Variants
        for i in range(AUGMENT_FACTOR):
            aug_seq = list(seq)
            
            # Apply random augmentations
            # A. Semantic Replacement (High value)
            if sim_dict:
                aug_seq = semantic_replace(aug_seq, sim_dict, prob=0.2)
            
            # B. Crop (Simulate shorter history)
            aug_seq = crop_sequence(aug_seq, prob=0.3)
            
            # C. Reorder (Robustness to noise)
            aug_seq = reorder_sequence(aug_seq, prob=0.3)
            
            if len(aug_seq) > 0:
                augmented_rows.append({
                    'user_id': user_id,
                    'history_item_id': str(aug_seq),
                    'type': 'augmented'
                })
                
    # Create new DataFrame
    aug_df = pd.DataFrame(augmented_rows)
    print(f"Original size: {len(df)}")
    print(f"Augmented size: {len(aug_df)}")
    
    print(f"Saving to {OUTPUT_FILE}...")
    aug_df.to_csv(OUTPUT_FILE, index=False)
    print("Done.")

if __name__ == "__main__":
    augment_data()
