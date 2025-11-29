"""
Transformer model for Chinese word segmentation using BMES tagging.
"""

import torch
import torch.nn as nn


class TransformerSeg(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2, num_classes=4, dropout=0.1, max_len=512):
        super(TransformerSeg, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 4, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        batch_size, seq_len = src.size()
        positions = torch.arange(seq_len, device=src.device).unsqueeze(0)
        
        x = self.dropout(self.embedding(src) + self.pos_embedding(positions))
        
        if src_mask is None:
            src_mask = (src == 0)

        output = self.transformer_encoder(x, src_key_padding_mask=src_mask)
        return self.fc_out(output)
