import torch
import pandas as pd
import ast
import numpy as np
from model import SASRec
from torch.utils.data import DataLoader, Dataset
import tqdm
import pickle
import os

# Configuration
BATCH_SIZE = 512
MAX_LEN = 100
HIDDEN_SIZE = 128
NUM_BLOCKS = 2
NUM_HEADS = 2
DROPOUT = 0.2
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
VOCAB_SIZE = 17408
TEXT_EMB_PATH = '/data/lzy/SASRec_Project/item_embeddings.pkl'
NGRAM_PATH = '/data/lzy/SASRec_Project/ngram_model_enhanced.pkl'
MODEL_PATH = 'sasrec_best.pth'

# Best Parameters found via Grid Search
SWITCH_THRESHOLD = 6
NGRAM_ORDER = 2

class EvaluationDataset(Dataset):
    def __init__(self, file_path, max_len=100, pad_token=0):
        self.max_len = max_len
        self.pad_token = pad_token
        
        print(f"Loading data from {file_path}...")
        self.df = pd.read_csv(file_path)
        self.sequences = []
        self.targets = []
        self.lengths = [] # Store actual lengths
        
        print("Parsing sequences for evaluation...")
        for seq_str in self.df['history_item_id']:
            try:
                seq = ast.literal_eval(seq_str)
            except:
                seq = []
            
            if len(seq) < 2:
                # Cannot evaluate if sequence length < 2 (need at least 1 input and 1 target)
                continue
                
            # Target is the last item
            target = seq[-1]
            # Input is everything before the last item
            input_seq = seq[:-1]
            
            # Truncate input if too long
            if len(input_seq) > max_len:
                input_seq = input_seq[-max_len:]
                
            self.sequences.append(input_seq)
            self.targets.append(target)
            self.lengths.append(len(input_seq))
        
        print(f"Loaded {len(self.sequences)} sequences for evaluation.")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        target = self.targets[idx]
        length = self.lengths[idx]
        
        # Left padding
        pad_len = self.max_len - len(seq)
        if pad_len > 0:
            padded_seq = [self.pad_token] * pad_len + seq
        else:
            padded_seq = seq[-self.max_len:]
            
        return torch.tensor(padded_seq, dtype=torch.long), torch.tensor(target, dtype=torch.long), torch.tensor(length, dtype=torch.long)

def evaluate():
    print(f"Using device: {DEVICE}")
    print(f"Using Hybrid Strategy: Threshold={SWITCH_THRESHOLD}, Order={NGRAM_ORDER}")
    
    # Load dataset
    dataset = EvaluationDataset('/data/lzy/SASRec_Project/test2.csv', max_len=MAX_LEN)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Load text embeddings
    print(f"Loading text embeddings from {TEXT_EMB_PATH}...")
    if os.path.exists(TEXT_EMB_PATH):
        with open(TEXT_EMB_PATH, 'rb') as f:
            text_embeddings = pickle.load(f)
        text_embeddings = text_embeddings.to(DEVICE)
    else:
        text_embeddings = None
    
    # Load N-Gram model
    print(f"Loading Enhanced N-Gram model from {NGRAM_PATH}...")
    if os.path.exists(NGRAM_PATH):
        with open(NGRAM_PATH, 'rb') as f:
            ngram_model = pickle.load(f)
    else:
        print("N-Gram model not found!")
        return
    
    # Load model
    model = SASRec(VOCAB_SIZE, HIDDEN_SIZE, NUM_BLOCKS, NUM_HEADS, MAX_LEN, DROPOUT, DEVICE, text_embeddings=text_embeddings).to(DEVICE)
    
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print(f"Loaded model weights from {MODEL_PATH}.")
    else:
        print(f"Model weights not found!")
        return

    model.eval()
    mrr_sum = 0.0
    count = 0
    
    print("Starting evaluation...")
    with torch.no_grad():
        for seq, target, length in tqdm.tqdm(loader):
            seq = seq.to(DEVICE)
            target = target.to(DEVICE)
            length = length.to(DEVICE)
            
            # Predict with SASRec
            logits = model.predict(seq) # [batch, vocab_size]
            logits[:, 0] = -float('inf')
            _, top_indices = torch.topk(logits, 10, dim=-1) # [batch, 10]
            
            # Convert to CPU for N-Gram processing
            seq_cpu = seq.cpu().numpy()
            length_cpu = length.cpu().numpy()
            top_indices_cpu = top_indices.cpu().numpy()
            
            final_top_indices = []
            
            batch_size = seq.size(0)
            for i in range(batch_size):
                l = length_cpu[i]
                
                # Hybrid Logic
                if l < SWITCH_THRESHOLD:
                    # Use N-Gram
                    ngram_top = []
                    
                    # Try 2-Gram first if enabled and possible
                    if NGRAM_ORDER == 2 and l >= 2:
                        prev_item = seq_cpu[i][-2]
                        curr_item = seq_cpu[i][-1]
                        key = (prev_item, curr_item)
                        
                        if key in ngram_model['2gram']:
                            ngram_preds = ngram_model['2gram'][key]
                            ngram_top = [x[0] for x in ngram_preds[:10]]
                    
                    # Fallback to 1-Gram if 2-Gram failed or not enabled
                    if not ngram_top:
                        curr_item = seq_cpu[i][-1]
                        if curr_item in ngram_model['1gram']:
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
                    
                    final_top_indices.append(ngram_top)
                else:
                    # Use SASRec
                    final_top_indices.append(top_indices_cpu[i])
            
            final_top_indices = torch.tensor(np.array(final_top_indices), device=DEVICE)
            
            # Calculate MRR
            targets_expanded = target.unsqueeze(1).expand_as(final_top_indices)
            hits = (final_top_indices == targets_expanded).nonzero(as_tuple=False)
            
            batch_mrrs = torch.zeros(seq.size(0), device=DEVICE)
            
            if hits.size(0) > 0:
                batch_indices = hits[:, 0]
                rank_indices = hits[:, 1]
                reciprocal_ranks = 1.0 / (rank_indices.float() + 1.0)
                batch_mrrs[batch_indices] = reciprocal_ranks
            
            mrr_sum += batch_mrrs.sum().item()
            count += seq.size(0)
            
    avg_mrr = mrr_sum / count
    print(f"Evaluation Results:")
    print(f"Total Sequences: {count}")
    print(f"MRR@10: {avg_mrr:.4f}")

if __name__ == "__main__":
    evaluate()
