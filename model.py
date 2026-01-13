import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SASRec(nn.Module):
    def __init__(self, vocab_size, hidden_size=64, num_blocks=2, num_heads=2, max_len=100, dropout=0.2, device='cuda', text_embeddings=None, use_fusion=True):
        super(SASRec, self).__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_len = max_len
        self.device = device
        self.num_heads = num_heads
        self.use_fusion = use_fusion
        
        self.item_emb = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        
        # Text embeddings integration
        self.text_embeddings = None
        if text_embeddings is not None:
            # text_embeddings: [vocab_size, text_emb_dim]
            self.text_embeddings = nn.Embedding.from_pretrained(text_embeddings, freeze=True, padding_idx=0)
            text_emb_dim = text_embeddings.size(1)
            self.text_proj = nn.Linear(text_emb_dim, hidden_size)
            
            # Deep Fusion Layer (Concat -> Linear -> ReLU)
            if use_fusion:
                self.fusion_layer = nn.Sequential(
                    nn.Linear(hidden_size * 2, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
            
        self.pos_emb = nn.Embedding(max_len, hidden_size)
        self.emb_dropout = nn.Dropout(dropout)
        
        # Use Pre-LN (norm_first=True) for better convergence
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, dropout=dropout, dim_feedforward=hidden_size*4, batch_first=True, norm_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_blocks)
        
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Skip text embeddings initialization as they are pretrained
            if self.text_embeddings is not None and module is self.text_embeddings:
                return
                
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, input_ids):
        # input_ids: [batch_size, seq_len]
        seq_len = input_ids.size(1)
        batch_size = input_ids.size(0)
        
        # Create attention mask (causal)
        # mask[i, j] = True if j > i
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=self.device), diagonal=1).bool()
        
        # Padding mask
        # key_padding_mask[b, j] = True if input_ids[b, j] == 0
        key_padding_mask = (input_ids == 0)
        
        # Combine masks to avoid NaN when a position is padded and has no valid attention target
        # We want:
        # 1. Causal masking: j > i -> Mask
        # 2. Padding masking: input_ids[j] == 0 -> Mask
        # 3. Exception: If input_ids[i] == 0 (it's a padded position), it must attend to SOMETHING to avoid NaN.
        #    We let it attend to itself.
        
        # Expand causal mask: (batch, seq_len, seq_len)
        extended_causal_mask = causal_mask.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Expand padding mask: (batch, 1, seq_len) -> broadcast to (batch, seq_len, seq_len)
        extended_padding_mask = key_padding_mask.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Combined mask
        combined_mask = extended_causal_mask | extended_padding_mask
        
        # Allow padded positions to attend to themselves
        # If input_ids[b, i] == 0, set combined_mask[b, i, i] = False
        # We can just set diagonal to False for ALL positions? 
        # No, causal mask already sets diagonal to False.
        # But padding mask might set it to True if input_ids[b, i] == 0.
        # So we force diagonal to False.
        
        ids = torch.arange(seq_len, device=self.device)
        combined_mask[:, ids, ids] = False
        
        # PyTorch Transformer expects (batch * num_heads, seq_len, seq_len) for 3D mask
        combined_mask = combined_mask.repeat_interleave(self.num_heads, dim=0)
        
        # Embeddings
        positions = torch.arange(seq_len, device=self.device).unsqueeze(0).expand(batch_size, -1)
        
        item_emb = self.item_emb(input_ids)
        
        if self.text_embeddings is not None:
            text_emb = self.text_embeddings(input_ids)
            text_emb = self.text_proj(text_emb)
            
            # Deep Fusion
            if self.use_fusion:
                combined = torch.cat([item_emb, text_emb], dim=-1)
                item_emb = self.fusion_layer(combined)
            else:
                item_emb = item_emb + text_emb
            
        x = item_emb + self.pos_emb(positions)
        x = self.emb_dropout(x)
        
        # Transformer
        # Pass combined_mask as mask, and None for src_key_padding_mask
        x = self.transformer_encoder(x, mask=combined_mask, src_key_padding_mask=None)
        
        # Zero out outputs for padded positions (just to be clean, though they shouldn't be NaN now)
        x = x * (torch.logical_not(key_padding_mask).unsqueeze(-1).float())
        
        x = self.layer_norm(x)
        
        return x

    def predict(self, input_ids):
        # Returns logits for the last position
        x = self.forward(input_ids)
        # Take the last step
        last_hidden = x[:, -1, :] # [batch_size, hidden_size]
        
        # Compute logits with all items
        # Reuse item embeddings
        # If using text embeddings, we should also include them in the output projection?
        # Standard SASRec uses item_emb.weight.
        # If we added text embeddings to input, the hidden state contains text info.
        # We can just dot product with item embeddings.
        # Or we can dot product with (item_emb + projected_text_emb).
        
        if self.text_embeddings is not None:
            # Get all item embeddings
            all_items = torch.arange(self.vocab_size, device=self.device)
            all_item_emb = self.item_emb(all_items)
            all_text_emb = self.text_embeddings(all_items)
            all_text_emb = self.text_proj(all_text_emb)
            
            # Deep Fusion for all items
            if self.use_fusion:
                combined = torch.cat([all_item_emb, all_text_emb], dim=-1)
                target_emb = self.fusion_layer(combined)
            else:
                target_emb = all_item_emb + all_text_emb
        else:
            target_emb = self.item_emb.weight
            
        logits = torch.matmul(last_hidden, target_emb.t()) # [batch_size, vocab_size]
        
        return logits

    def calculate_cl_loss(self, seq1, seq2):
        # seq1, seq2: [batch, seq_len]
        # Get sequence representations (last hidden state)
        
        feat1 = self.forward(seq1) # [batch, seq_len, hidden]
        feat2 = self.forward(seq2) # [batch, seq_len, hidden]
        
        # Use the last position as sequence representation
        emb1 = feat1[:, -1, :] # [batch, hidden]
        emb2 = feat2[:, -1, :] # [batch, hidden]
        
        # Normalize
        emb1 = F.normalize(emb1, dim=1)
        emb2 = F.normalize(emb2, dim=1)
        
        # InfoNCE Loss
        temperature = 0.1
        
        # Positive pairs: (emb1[i], emb2[i])
        pos_score = torch.sum(emb1 * emb2, dim=1) / temperature
        pos_score = torch.exp(pos_score)
        
        # Negative pairs: (emb1[i], emb2[j]) for all j
        # Similarity matrix: [batch, batch]
        sim_matrix = torch.matmul(emb1, emb2.t()) / temperature
        neg_score = torch.sum(torch.exp(sim_matrix), dim=1)
        
        # Loss = -log(pos / sum(neg))
        loss = -torch.log(pos_score / (neg_score + 1e-8))
        return loss.mean()
