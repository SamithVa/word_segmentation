"""
Transformer model for Chinese word segmentation using BMES tagging.
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Implement the absolute positional encoding as described in "Attention is All You Need"."""
    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()
        # Create a matrix of [max_len, d_model] representing positions
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sin to even indices, cos to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a learnable parameter, but part of state_dict)
        self.register_buffer('pe', pe.unsqueeze(0))
 
    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        # Add positional encoding to embeddings
        x = x + self.pe[:, :x.size(1), :]
        return x


class TransformerSeg(nn.Module):
    """Transformer Encoder segmenter for Chinese word segmentation."""

    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2, num_classes=4, dropout=0.1):
        super(TransformerSeg, self).__init__()
        
        # 1. Embedding Layer
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # 2. Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # 3. Transformer Encoder
        # batch_first=True ensures input format is [batch, seq_len, feature]
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. Output Layer (Project to B, M, E, S)
        self.fc_out = nn.Linear(d_model, num_classes)
        
        self.d_model = d_model

    def forward(self, src, src_mask=None):
        # src: [batch_size, seq_len]
        
        # Create padding mask if not provided
        if src_mask is None:
            src_mask = (src == 0)
        
        # Embed and scale (scaling by sqrt(d_model) is a Transformer best practice)
        x = self.embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        
        # Pass through Transformer
        # output: [batch_size, seq_len, d_model]
        output = self.transformer_encoder(x, src_key_padding_mask=src_mask)
        
        # Project to classes
        # output: [batch_size, seq_len, 4]
        logits = self.fc_out(output)
        return logits