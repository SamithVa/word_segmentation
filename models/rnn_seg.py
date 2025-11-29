"""
RNN model for Chinese word segmentation using BMES tagging.
"""

import torch.nn as nn


class RNNSeg(nn.Module):
    """Vanilla RNN segmenter for Chinese word segmentation."""

    def __init__(self, vocab_size, d_model=128, hidden_dim=256, num_layers=2, num_classes=4):
        super(RNNSeg, self).__init__()
        
        # 1. Embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # 2. Vanilla RNN Layer
        # batch_first=True ensures input is [batch, seq_len, feature]
        # bidirectional=False (Standard RNN scans left->right only)
        self.rnn = nn.RNN(input_size=d_model, 
                          hidden_size=hidden_dim, 
                          num_layers=num_layers, 
                          batch_first=True,
                          bidirectional=False) 
        
        # 3. Output Layer
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: [batch, seq_len]
        embeds = self.embedding(x)
        
        # rnn_out: [batch, seq_len, hidden_dim]
        # h_n: [num_layers, batch, hidden_dim] (hidden state)
        rnn_out, h_n = self.rnn(embeds)
        
        # Project to BMES classes
        logits = self.fc(rnn_out)
        return logits
