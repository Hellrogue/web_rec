import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import SASRecDataset
from model import SASRec
import numpy as np
import pandas as pd
import os
import sys
import pickle

# Configuration
BATCH_SIZE = 64
LR = 0.001
EPOCHS = 5 # Increased slightly for convergence
MAX_LEN = 100
HIDDEN_SIZE = 256
NUM_BLOCKS = 4 # Deeper model
NUM_HEADS = 4
DROPOUT = 0.1 # Reduced dropout
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
VOCAB_SIZE = 17408
TEXT_EMB_PATH = '/data/lzy/SASRec_Project/item_embeddings.pkl'
CL_WEIGHT = 0.1 # Weight for Contrastive Learning Loss

def augment_sequence(seq, mask_prob=0.2):
    # seq: [batch, len]
    # Randomly mask items with 0 (padding)
    # Create a mask of same shape
    mask = torch.rand_like(seq.float()) < mask_prob
    # Don't mask padding (0)
    mask = mask & (seq != 0)
    
    aug_seq = seq.clone()
    aug_seq[mask] = 0
    return aug_seq

def train():
    print(f"Using device: {DEVICE}")
    
    # Load dataframe
    print("Loading dataframe...")
    # Use Augmented Dataset
    df = pd.read_csv('/data/lzy/SASRec_Project/train_augmented.csv')
    
    # Split train/val manually
    # Shuffle dataframe
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    val_size = int(len(df) * 0.05)
    train_size = len(df) - val_size
    
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:]
    
    print(f"Train size: {len(train_df)}, Val size: {len(val_df)}")
    
    # Create datasets
    # Enable augmentation for training
    # mask_prob=0.0: Don't mask main training data to preserve supervision
    # sliding_window=True: Use exhaustive sliding window (stride=1)
    # Note: Since we already augmented offline, we can reduce online augmentation or keep it for even more diversity.
    # Let's keep sliding_window=True to get all subsequences from the augmented sequences.
    train_dataset = SASRecDataset(
        train_df, 
        max_len=MAX_LEN, 
        augment=False, # Disable online augmentation as we did it offline
        mask_prob=0.0, 
        sliding_window=True,
        crop_prob=0.0,
        reorder_prob=0.0, 
        semantic_prob=0.0,
        similarity_path='/data/lzy/SASRec_Project/item_similarity.pkl'
    )
    val_dataset = SASRecDataset(val_df, max_len=MAX_LEN, augment=False, sliding_window=False)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Load text embeddings
    print(f"Loading text embeddings from {TEXT_EMB_PATH}...")
    with open(TEXT_EMB_PATH, 'rb') as f:
        text_embeddings = pickle.load(f)
    text_embeddings = text_embeddings.to(DEVICE)
    
    # Model
    model = SASRec(VOCAB_SIZE, HIDDEN_SIZE, NUM_BLOCKS, NUM_HEADS, MAX_LEN, DROPOUT, DEVICE, text_embeddings=text_embeddings).to(DEVICE)
    
    # Label Smoothing helps lower the loss value and improves generalization
    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1) 
    
    # AdamW with Weight Decay
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    
    # OneCycleLR for Super Convergence
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR, steps_per_epoch=len(train_loader), epochs=EPOCHS)
    
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for batch_idx, seq in enumerate(train_loader):
            seq = seq.to(DEVICE) # [batch, max_len]
            
            # Input: [s_1, ..., s_{L-1}]
            input_ids = seq[:, :-1]
            # Target: [s_2, ..., s_L]
            target_ids = seq[:, 1:]
            
            optimizer.zero_grad()
            
            # Forward
            # Output: [batch, max_len-1, hidden_size]
            outputs = model(input_ids)
            
            # Contrastive Learning Task
            # Generate two augmented views
            aug_seq1 = augment_sequence(seq, mask_prob=0.2)
            aug_seq2 = augment_sequence(seq, mask_prob=0.2)
            
            cl_loss = model.calculate_cl_loss(aug_seq1, aug_seq2)
            
            # Reshape for loss
            # outputs: [batch * (max_len-1), hidden_size] -> logits: [batch * (max_len-1), vocab_size]
            
            # Compute logits
            # We need to use the same projection as in model.predict
            if model.text_embeddings is not None:
                all_items = torch.arange(VOCAB_SIZE, device=DEVICE)
                all_item_emb = model.item_emb(all_items)
                all_text_emb = model.text_embeddings(all_items)
                all_text_emb = model.text_proj(all_text_emb)
                
                # Deep Fusion
                combined = torch.cat([all_item_emb, all_text_emb], dim=-1)
                target_emb = model.fusion_layer(combined)
            else:
                target_emb = model.item_emb.weight
            
            # Experiment: Only use learned item embeddings for output projection
            # target_emb = model.item_emb.weight
                
            logits = torch.matmul(outputs, target_emb.t()) # [batch, len, vocab]
            
            rec_loss = criterion(logits.view(-1, VOCAB_SIZE), target_ids.reshape(-1))
            
            # Total Loss
            loss = rec_loss + CL_WEIGHT * cl_loss
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}/{EPOCHS}, Batch {batch_idx}, Loss: {loss.item():.4f}, Rec: {rec_loss.item():.4f}, CL: {cl_loss.item():.4f}")
        
            scheduler.step() # Step scheduler every batch for OneCycleLR
            
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} Training Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            # Precompute target embeddings for validation
            if model.text_embeddings is not None:
                all_items = torch.arange(VOCAB_SIZE, device=DEVICE)
                all_item_emb = model.item_emb(all_items)
                all_text_emb = model.text_embeddings(all_items)
                all_text_emb = model.text_proj(all_text_emb)
                
                # Deep Fusion
                combined = torch.cat([all_item_emb, all_text_emb], dim=-1)
                target_emb = model.fusion_layer(combined)
            else:
                target_emb = model.item_emb.weight
            
            # Experiment: Only use learned item embeddings for output projection
            # target_emb = model.item_emb.weight
                
            for seq in val_loader:
                seq = seq.to(DEVICE)
                input_ids = seq[:, :-1]
                target_ids = seq[:, 1:]
                
                outputs = model(input_ids)
                logits = torch.matmul(outputs, target_emb.t())
                loss = criterion(logits.view(-1, VOCAB_SIZE), target_ids.reshape(-1))
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1} Validation Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'sasrec_best.pth')
            print("Saved best model.")

if __name__ == "__main__":
    train()
