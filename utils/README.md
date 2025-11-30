# Shared Utilities for Chinese Word Segmentation

Utility modules for the segmentation pipeline.

## Modules

| Module | Description |
|--------|-------------|
| `dataset.py` | `ChineseSegDataset`, `collate_fn`, `build_vocab` |
| `trainer.py` | `train_epoch`, `evaluate`, `train_model` |
| `tokenizer.py` | `BaseTokenizer` for BMES to words conversion |
| `evaluation.py` | `evaluate_segmentation`, `evaluate_model_on_all_datasets` |

## Usage

```python
from utils import ChineseSegDataset, build_vocab, collate_fn, train_epoch, evaluate

# Build vocabulary
char2idx, tag2idx = build_vocab(['training.txt'], min_freq=2)

# Create dataset and dataloader
dataset = ChineseSegDataset('training.txt', char2idx, tag2idx, max_len=256)
loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)

# Training
train_loss, train_acc = train_epoch(model, loader, criterion, optimizer, device)
val_loss, val_acc = evaluate(model, val_loader, criterion, device)
```

## BMES Tagging

- **B**: Begin of multi-char word
- **M**: Middle of multi-char word  
- **E**: End of multi-char word
- **S**: Single character word
