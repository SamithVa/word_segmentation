"""
BiLSTM model for Chinese word segmentation using BMES tagging.
"""

import torch.nn as nn


class LSTMSeg(nn.Module):
    """Bidirectional LSTM segmenter for Chinese word segmentation."""

    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2, num_classes=4, dropout=0.1):
        super(LSTMSeg, self).__init__()
        
        # 1. Embedding Layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # 2. BiLSTM Layer
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 3. Output Layer (Project to B, I, E, S)
        # BiLSTM outputs hidden_dim * 2 (forward + backward)
        self.fc_out = nn.Linear(hidden_dim * 2, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src: [batch_size, seq_len]
        
        # Embed
        # x: [batch_size, seq_len, embedding_dim]
        x = self.embedding(src)
        x = self.dropout(x)
        
        # Pass through BiLSTM
        # lstm_out: [batch_size, seq_len, hidden_dim * 2]
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        
        # Project to classes
        # logits: [batch_size, seq_len, num_classes]
        logits = self.fc_out(lstm_out)
        
        return logits
