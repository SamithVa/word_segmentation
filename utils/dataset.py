"""
Dataset utilities for Chinese word segmentation.
"""

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
import os


class ChineseSegDataset(Dataset):
    def __init__(self, filepath, char2idx, tag2idx, max_len=512):
        """
        Load and prepare Chinese word segmentation dataset.
        
        Args:
            filepath: path to training file (space-separated words)
            char2idx: character to index mapping
            tag2idx: tag to index mapping (BMES)
            max_len: maximum sequence length
        """
        self.char2idx = char2idx
        self.tag2idx = tag2idx
        self.max_len = max_len
        self.samples = []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                words = line.strip().split()
                if not words:
                    continue
                
                # Convert words to characters and BMES tags
                chars = []
                tags = []
                for word in words:
                    word_chars = list(word)
                    chars.extend(word_chars)
                    
                    if len(word) == 1:
                        tags.append('S')
                    else:
                        tags.append('B')
                        tags.extend(['M'] * (len(word) - 2))
                        tags.append('E')
                
                # Truncate if too long
                if len(chars) > max_len:
                    chars = chars[:max_len]
                    tags = tags[:max_len]
                
                # Convert to indices
                char_ids = [self.char2idx.get(c, self.char2idx['<UNK>']) for c in chars]
                tag_ids = [self.tag2idx[t] for t in tags]
                
                self.samples.append((char_ids, tag_ids))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        char_ids, tag_ids = self.samples[idx]
        return torch.tensor(char_ids, dtype=torch.long), torch.tensor(tag_ids, dtype=torch.long)


def collate_fn(batch):
    """
    Pad sequences to same length in a batch.
    
    Args:
        batch: list of (chars, tags) tuples
        
    Returns:
        chars_padded: [batch_size, max_seq_len]
        tags_padded: [batch_size, max_seq_len]
    """
    chars, tags = zip(*batch)
    
    # Pad sequences (padding value = 0)
    chars_padded = pad_sequence(chars, batch_first=True, padding_value=0)
    tags_padded = pad_sequence(tags, batch_first=True, padding_value=0)
    
    return chars_padded, tags_padded


def build_vocab(filepaths, min_freq=2):
    """
    Build character vocabulary from training files.
    
    Args:
        filepaths: list of training file paths
        min_freq: minimum frequency for a character to be included
        
    Returns:
        char2idx: character to index mapping
        tag2idx: tag to index mapping (BMES)
    """
    char_counter = Counter()
    
    for filepath in filepaths:
        if not os.path.exists(filepath):
            continue
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                words = line.strip().split()
                for word in words:
                    char_counter.update(word)
    
    # Build vocab: <PAD>=0, <UNK>=1, then frequent chars
    char2idx = {'<PAD>': 0, '<UNK>': 1}
    for char, freq in char_counter.items():
        if freq >= min_freq:
            char2idx[char] = len(char2idx)
    
    # Tag vocabulary (BMES)
    tag2idx = {'<PAD>': 0, 'B': 1, 'M': 2, 'E': 3, 'S': 4}
    
    return char2idx, tag2idx
