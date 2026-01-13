import torch
from torch.utils.data import Dataset
import pandas as pd
import ast
import numpy as np
import random
import pickle

class SASRecDataset(Dataset):
    def __init__(self, data, max_len=100, pad_token=0, augment=False, mask_prob=0.0, sliding_window=False, min_len=3, 
                 crop_prob=0.0, reorder_prob=0.0, semantic_prob=0.0, similarity_path=None):
        self.max_len = max_len
        self.pad_token = pad_token
        self.augment = augment
        self.mask_prob = mask_prob
        self.sliding_window = sliding_window
        self.min_len = min_len
        self.crop_prob = crop_prob
        self.reorder_prob = reorder_prob
        self.semantic_prob = semantic_prob
        
        # Load similarity dictionary if semantic augmentation is enabled
        self.similarity_dict = None
        if self.augment and self.semantic_prob > 0 and similarity_path:
            print(f"Loading similarity dictionary from {similarity_path}...")
            with open(similarity_path, 'rb') as f:
                self.similarity_dict = pickle.load(f)
        
        if isinstance(data, str):
            print(f"Loading data from {data}...")
            self.df = pd.read_csv(data)
        elif isinstance(data, pd.DataFrame):
            self.df = data
        else:
            raise ValueError("Data must be file path or DataFrame")
        
        self.users = self.df['user_id'].values
        self.sequences = []
        
        print("Parsing sequences...")
        for seq_str in self.df['history_item_id']:
            try:
                if isinstance(seq_str, str):
                    seq = ast.literal_eval(seq_str)
                else:
                    seq = seq_str # Already list
                
                if self.sliding_window:
                    # Adaptive Sliding Window
                    seq_len = len(seq)
                    
                    # Strategy:
                    # Always use stride=1 to maximize data usage (Data Augmentation)
                    stride = 1
                        
                    # Generate subsequences
                    for i in range(self.min_len, seq_len + 1, stride):
                        sub_seq = seq[:i]
                        if len(sub_seq) > self.max_len:
                            sub_seq = sub_seq[-self.max_len:]
                        self.sequences.append(sub_seq)
                        
                    # Ensure the very last sequence is included if stride skipped it
                    if (seq_len - self.min_len) % stride != 0:
                        sub_seq = seq
                        if len(sub_seq) > self.max_len:
                            sub_seq = sub_seq[-self.max_len:]
                        self.sequences.append(sub_seq)
                        
                else:
                    # Truncate if too long (keep recent items)
                    if len(seq) > max_len:
                        seq = seq[-max_len:]
                    self.sequences.append(seq)
            except:
                if not self.sliding_window:
                    self.sequences.append([])
                
        print(f"Loaded {len(self.sequences)} sequences.")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = list(self.sequences[idx]) # Copy to avoid modifying original
        
        if self.augment:
            # 1. Crop (Drop beginning)
            if self.crop_prob > 0 and random.random() < self.crop_prob and len(seq) > 1:
                # Drop random amount from start, keep at least 1 item
                # Max drop is len(seq) // 2
                max_drop = len(seq) // 2
                if max_drop > 0:
                    drop_len = random.randint(1, max_drop)
                    seq = seq[drop_len:]
            
            # 2. Reorder (Shuffle sub-window)
            if self.reorder_prob > 0 and random.random() < self.reorder_prob and len(seq) > 3:
                # Pick a random window of size 3
                window_size = 3
                start_idx = random.randint(0, len(seq) - window_size)
                sub_window = seq[start_idx : start_idx + window_size]
                random.shuffle(sub_window)
                seq[start_idx : start_idx + window_size] = sub_window
                
            # 3. Semantic Replacement
            if self.semantic_prob > 0 and self.similarity_dict is not None:
                new_seq = []
                for item in seq:
                    if item in self.similarity_dict and random.random() < self.semantic_prob:
                        # Replace with similar item
                        similar_items = self.similarity_dict[item]
                        if similar_items:
                            new_seq.append(random.choice(similar_items))
                        else:
                            new_seq.append(item)
                    else:
                        new_seq.append(item)
                seq = new_seq
                
            # 4. Masking
            if self.mask_prob > 0:
                new_seq = []
                for item in seq:
                    if random.random() < self.mask_prob:
                        new_seq.append(self.pad_token)
                    else:
                        new_seq.append(item)
                seq = new_seq
        
        # Left padding
        pad_len = self.max_len - len(seq)
        if pad_len > 0:
            padded_seq = [self.pad_token] * pad_len + seq
        else:
            padded_seq = seq[-self.max_len:]
            
        return torch.tensor(padded_seq, dtype=torch.long)
