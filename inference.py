import torch
import pandas as pd
import ast
import numpy as np
from model import SASRec
from torch.utils.data import DataLoader, Dataset
import csv
import sys
import pickle
import datetime
import argparse
import os

# Configuration (Must match train.py)
BATCH_SIZE = 512
MAX_LEN = 100
HIDDEN_SIZE = 128
NUM_BLOCKS = 2
NUM_HEADS = 2
DROPOUT = 0.1
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
VOCAB_SIZE = 17408
TEXT_EMB_PATH = '/data/lzy/SASRec_Project/item_embeddings.pkl'
NGRAM_PATH = '/data/lzy/SASRec_Project/ngram_model_enhanced.pkl'
MODEL_PATH = 'sasrec_best.pth'

# Best Parameters
SWITCH_THRESHOLD = 0 # Disable N-Gram to test pure SASRec performance
NGRAM_ORDER = 2

class InferenceDataset(Dataset):
    def __init__(self, file_path, max_len=100, pad_token=0):
        self.max_len = max_len
        self.pad_token = pad_token
        
        print(f"Loading data from {file_path}...")
        self.df = pd.read_csv(file_path)
        self.users = self.df['user_id'].values
        self.sequences = []
        self.lengths = []
        
        print("Parsing sequences...")
        for seq_str in self.df['history_item_id']:
            try:
                seq = ast.literal_eval(seq_str)
                self.lengths.append(len(seq))
                if len(seq) > max_len:
                    seq = seq[-max_len:]
                self.sequences.append(seq)
            except:
                self.sequences.append([])
                self.lengths.append(0)
                
    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        user_id = self.users[idx]
        length = self.lengths[idx]
        
        # Left padding
        pad_len = self.max_len - len(seq)
        if pad_len > 0:
            padded_seq = [self.pad_token] * pad_len + seq
        else:
            padded_seq = seq[-self.max_len:]
            
        return torch.tensor(padded_seq, dtype=torch.long), user_id, torch.tensor(length, dtype=torch.long)

def generate_submission(score=None):
    print(f"Using device: {DEVICE}")
    print(f"Using Hybrid Strategy: Threshold={SWITCH_THRESHOLD}, Order={NGRAM_ORDER}")
    
    # Load dataset
    dataset = InferenceDataset('/data/lzy/SASRec_Project/test2.csv', max_len=MAX_LEN)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Load text embeddings
    print(f"Loading text embeddings from {TEXT_EMB_PATH}...")
    if os.path.exists(TEXT_EMB_PATH):
        with open(TEXT_EMB_PATH, 'rb') as f:
            text_embeddings = pickle.load(f)
        text_embeddings = text_embeddings.to(DEVICE)
    else:
        print("Text embeddings not found, initializing randomly.")
        text_embeddings = None
    # text_embeddings = None # Force None to match checkpoint structure
        
    # Load N-Gram model
    print(f"Loading Enhanced N-Gram model from {NGRAM_PATH}...")
    if os.path.exists(NGRAM_PATH):
        with open(NGRAM_PATH, 'rb') as f:
            ngram_model = pickle.load(f)
    else:
        print("N-Gram model not found! Will use SASRec for all.")
        ngram_model = {}
    
    # Load model
    model = SASRec(VOCAB_SIZE, HIDDEN_SIZE, NUM_BLOCKS, NUM_HEADS, MAX_LEN, DROPOUT, DEVICE, text_embeddings=text_embeddings, use_fusion=False).to(DEVICE)
    
    # Load weights
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print(f"Loaded model weights from {MODEL_PATH}.")
    else:
        print(f"Model weights not found at {MODEL_PATH}! Please train the model first.")
        return

    model.eval()
    
    results = []
    
    print("Generating predictions...")
    with torch.no_grad():
        for seq, user_ids, lengths in loader:
            seq = seq.to(DEVICE)
            lengths = lengths.to(DEVICE)
            
            # Predict with SASRec
            logits = model.predict(seq) # [batch, vocab_size]
            
            # Set padding index (0) logits to -inf
            logits[:, 0] = -float('inf')
            
            # Get Top-10 from SASRec
            _, top_indices = torch.topk(logits, 10, dim=-1) # [batch, 10]
            
            # Convert to CPU
            seq_cpu = seq.cpu().numpy()
            length_cpu = lengths.cpu().numpy()
            top_indices_cpu = top_indices.cpu().numpy()
            user_ids_cpu = user_ids.numpy()
            
            batch_size = seq.size(0)
            
            for i in range(batch_size):
                user_id = user_ids_cpu[i]
                l = length_cpu[i]
                
                final_top = []
                
                if l < SWITCH_THRESHOLD and ngram_model:
                    # Use N-Gram
                    ngram_top = []
                    
                    # Try 2-Gram first if enabled and possible
                    if NGRAM_ORDER == 2 and l >= 2:
                        prev_item = seq_cpu[i][-2]
                        curr_item = seq_cpu[i][-1]
                        key = (prev_item, curr_item)
                        
                        if '2gram' in ngram_model and key in ngram_model['2gram']:
                            ngram_preds = ngram_model['2gram'][key]
                            ngram_top = [x[0] for x in ngram_preds[:10]]
                    
                    # Fallback to 1-Gram if 2-Gram failed or not enabled
                    if not ngram_top:
                        curr_item = seq_cpu[i][-1]
                        if '1gram' in ngram_model and curr_item in ngram_model['1gram']:
                            ngram_preds = ngram_model['1gram'][curr_item]
                            ngram_top = [x[0] for x in ngram_preds[:10]]
                    
                    # Fill with SASRec if needed
                    if len(ngram_top) < 10:
                        sasrec_top = top_indices_cpu[i]
                        for item in sasrec_top:
                            if item not in ngram_top:
                                ngram_top.append(item)
                            if len(ngram_top) == 10:
                                break
                    
                    final_top = ngram_top
                else:
                    final_top = top_indices_cpu[i].tolist()
                
                # Format for submission
                # user_id,"[item1, item2, ..., item10]"
                # The target format requires a list string enclosed in quotes
                pred_list_str = str(final_top) # This converts [1, 2] to "[1, 2]"
                results.append(f'{user_id},"{pred_list_str}"')
    
    # Save submission
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"submission_{timestamp}.csv"
    if score:
        filename = f"submission_{score}_{timestamp}.csv"
        
    with open(filename, 'w') as f:
        f.write("user_id,item_id\n")
        for line in results:
            f.write(line + "\n")
            
    print(f"Saved submission to {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--score", type=str, help="Optional score to append to filename")
    args = parser.parse_args()
    
    generate_submission(args.score)
