import pandas as pd
import ast
import pickle
from collections import defaultdict, Counter
import tqdm
import os

# Configuration
DATA_PATH = '/data/lzy/SASRec_Project/test2.csv'
OUTPUT_PATH = '/data/lzy/SASRec_Project/ngram_model_enhanced.pkl'

def build_ngram():
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    
    # 1-Gram: P(next | current)
    transitions_1gram = defaultdict(Counter)
    # 2-Gram: P(next | prev, current)
    transitions_2gram = defaultdict(Counter)
    
    print("Counting transitions...")
    for seq_str in tqdm.tqdm(df['history_item_id']):
        try:
            seq = ast.literal_eval(seq_str)
            if len(seq) < 2:
                continue
                
            # Iterate through sequence
            for i in range(len(seq) - 1):
                current_item = seq[i]
                next_item = seq[i+1]
                
                # 1-Gram
                transitions_1gram[current_item][next_item] += 1
                
                # 2-Gram
                if i > 0:
                    prev_item = seq[i-1]
                    key = (prev_item, current_item)
                    transitions_2gram[key][next_item] += 1
                    
        except:
            continue
            
    print(f"Captured 1-gram transitions for {len(transitions_1gram)} items.")
    print(f"Captured 2-gram transitions for {len(transitions_2gram)} pairs.")
    
    ngram_model = {
        '1gram': {},
        '2gram': {}
    }
    
    print("Processing 1-gram transitions...")
    for item, counts in tqdm.tqdm(transitions_1gram.items()):
        total = sum(counts.values())
        probs = [(next_item, count / total) for next_item, count in counts.most_common(20)] # Keep top 20
        ngram_model['1gram'][item] = probs

    print("Processing 2-gram transitions...")
    for pair, counts in tqdm.tqdm(transitions_2gram.items()):
        total = sum(counts.values())
        probs = [(next_item, count / total) for next_item, count in counts.most_common(20)] # Keep top 20
        ngram_model['2gram'][pair] = probs
        
    print(f"Saving Enhanced N-Gram model to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, 'wb') as f:
        pickle.dump(ngram_model, f)
        
    print("Done.")

if __name__ == "__main__":
    build_ngram()
