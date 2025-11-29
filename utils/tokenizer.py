"""
Base tokenizer for word segmentation inference.
"""

import torch


class BaseTokenizer:
    """Base class for tokenizers that convert BMES tags to words."""
    
    def __init__(self, model, char2idx, idx2tag, device):
        """
        Initialize tokenizer.
        
        Args:
            model: trained model
            char2idx: character to index mapping
            idx2tag: index to tag mapping
            device: torch device
        """
        self.model = model
        self.char2idx = char2idx
        self.idx2tag = idx2tag
        self.device = device

        # Only call eval() for PyTorch models
        if hasattr(model, 'eval') and callable(getattr(model, 'eval')):
            self.model.eval()
    
    def tokenize(self, text, max_len=512):
        """
        Tokenize a single sentence.
        
        Args:
            text: input Chinese text
            max_len: maximum sequence length
            
        Returns:
            words: list of segmented words
        """
        # Convert characters to indices
        # Handle UNK token gracefully for models that don't have it
        unk_idx = self.char2idx.get('<UNK>', 0)  # Default to 0 if UNK not found
        char_ids = [self.char2idx.get(c, unk_idx) for c in text]
        
        # Truncate if needed
        if len(char_ids) > max_len:
            char_ids = char_ids[:max_len]
            text = text[:max_len]
        
        # Convert to tensor
        chars_tensor = torch.tensor([char_ids], dtype=torch.long).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(chars_tensor)  # [1, seq_len, num_classes]
            predictions = outputs.argmax(dim=-1).squeeze(0).cpu().tolist()
        
        # Convert predictions to tags
        tags = [self.idx2tag[pred] for pred in predictions[:len(text)]]
        
        # Convert tags to words
        return self._tags_to_words(text, tags)
    
    def _tags_to_words(self, text, tags):
        """
        Convert BMES tags to words.
        
        Args:
            text: input text
            tags: list of BMES tags
            
        Returns:
            words: list of words
        """
        words = []
        word = ""
        for char, tag in zip(text, tags):
            if tag == 'B':
                if word:
                    words.append(word)
                word = char
            elif tag == 'M':
                word += char
            elif tag == 'E':
                word += char
                words.append(word)
                word = ""
            elif tag == 'S':
                if word:
                    words.append(word)
                words.append(char)
                word = ""
        
        if word:
            words.append(word)
        
        return words
