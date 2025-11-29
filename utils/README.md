# Shared Utilities for Chinese Word Segmentation

This directory contains reusable utility modules for Chinese word segmentation tasks. These modules are designed to work across different model architectures (RNN, LSTM, Bi-LSTM, Transformer, etc.).

## Modules

### `dataset.py`
Dataset and vocabulary utilities.

**Classes:**
- `ChineseSegDataset`: PyTorch Dataset for loading Chinese segmentation data with BMES tagging

**Functions:**
- `collate_fn(batch)`: Batch collation with padding
- `build_vocab(filepaths, min_freq=2)`: Build character and tag vocabularies

**Usage:**
```python
from utils import ChineseSegDataset, build_vocab, collate_fn

# Build vocabulary
char2idx, tag2idx = build_vocab(['training.txt'], min_freq=2)

# Create dataset
dataset = ChineseSegDataset('training.txt', char2idx, tag2idx, max_len=256)

# Create dataloader
from torch.utils.data import DataLoader
loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)
```

### `trainer.py`
Training and evaluation functions.

**Functions:**
- `train_epoch(model, dataloader, criterion, optimizer, device)`: Train for one epoch
- `evaluate(model, dataloader, criterion, device)`: Evaluate model performance

**Usage:**
```python
from utils import train_epoch, evaluate

# Training loop
for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
```

### `tokenizer.py`
Base tokenizer class for inference.

**Classes:**
- `BaseTokenizer`: Base class for converting BMES predictions to words

**Usage:**
```python
from utils import BaseTokenizer

class MyTokenizer(BaseTokenizer):
    """Model-specific tokenizer that inherits from BaseTokenizer."""
    pass

tokenizer = MyTokenizer(model, char2idx, idx2tag, device)
words = tokenizer.tokenize("中文分词示例")
```

### `evaluation.py`
Evaluation metrics for word segmentation.

**Functions:**
- `evaluate_segmentation(tokenizer, gold_filepath, test_filepath)`: Calculate precision, recall, and F1 score

**Usage:**
```python
from utils import evaluate_segmentation

tokenizer = MyTokenizer(model, char2idx, idx2tag, device)
results = evaluate_segmentation(tokenizer, 'gold.txt', 'test.txt')
print(f"F1: {results['f1']:.4f}")
```

### `visualization.py`
Visualization utilities for training metrics.

**Functions:**
- `plot_training_history(train_losses, train_accuracies, output_path)`: Plot training curves

**Usage:**
```python
from utils import plot_training_history

train_losses = [0.5, 0.4, 0.3, 0.2]
train_accs = [0.85, 0.88, 0.91, 0.93]
plot_training_history(train_losses, train_accs, 'training.png')
```

## Integration Example

Here's a complete example showing how to use all utilities together:

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils import (
    ChineseSegDataset,
    collate_fn,
    build_vocab,
    train_epoch,
    evaluate,
    BaseTokenizer,
    evaluate_segmentation,
    plot_training_history
)

# Build vocabulary
training_files = ['train1.txt', 'train2.txt']
char2idx, tag2idx = build_vocab(training_files, min_freq=2)
idx2tag = {v: k for k, v in tag2idx.items()}

# Create datasets
datasets = [ChineseSegDataset(f, char2idx, tag2idx) for f in training_files]
train_dataset = ConcatDataset(datasets)
train_loader = DataLoader(train_dataset, batch_size=64, collate_fn=collate_fn)

# Initialize model (your custom model)
model = YourModel(vocab_size=len(char2idx), num_classes=len(tag2idx))
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
train_losses = []
train_accs = []
for epoch in range(10):
    loss, acc = train_epoch(model, train_loader, criterion, optimizer, device)
    train_losses.append(loss)
    train_accs.append(acc)

# Plot results
plot_training_history(train_losses, train_accs, 'training_history.png')

# Evaluation
class MyTokenizer(BaseTokenizer):
    pass

tokenizer = MyTokenizer(model, char2idx, idx2tag, device)
results = evaluate_segmentation(tokenizer, 'gold.txt', 'test.txt')
print(f"F1 Score: {results['f1']:.4f}")
```

## Design Philosophy

1. **Modularity**: Each module handles a specific aspect of the pipeline
2. **Reusability**: Common code shared across different implementations
3. **Extensibility**: Easy to extend (e.g., inherit from BaseTokenizer)
4. **Consistency**: Same interface across different models
5. **Maintainability**: Bug fixes in one place benefit all implementations

## BMES Tagging Scheme

All utilities use the BMES tagging scheme for word segmentation:

- **B** (Begin): First character of a multi-character word
- **M** (Middle): Middle characters of a multi-character word
- **E** (End): Last character of a multi-character word
- **S** (Single): Single character word
- **<PAD>**: Padding token (index 0)

## Benefits of Using These Utilities

1. **Consistency**: Same preprocessing and evaluation across all models
2. **Fair Comparison**: Models use identical data loading and evaluation
3. **Less Code Duplication**: Write once, use everywhere
4. **Easier Maintenance**: Update once, fix everywhere
5. **Faster Development**: Focus on model architecture, not boilerplate

## Future Extensions

Potential additions to consider:

- Data augmentation utilities
- Model checkpointing helpers
- Learning rate scheduling utilities
- Mixed precision training support
- Multi-GPU training utilities
- Custom evaluation metrics (OOV accuracy, etc.)
