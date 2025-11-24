import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import math
import os
import pickle
from collections import Counter
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

class PositionalEncoding(nn.Module):
    """Implement the absolute positional encoding as described in "Attention is All You Need"."""
    def __init__(self, d_model, max_len=5000):
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

class TransformerSegModel(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2, num_classes=4, dropout=0.1):
        super(TransformerSegModel, self).__init__()
        
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

class ChineseSegDataset(Dataset):
    def __init__(self, filepath, char2idx, tag2idx, max_len=512):
        """
        Load and prepare Chinese word segmentation dataset.
        filepath: path to training file (space-separated words)
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
    """Pad sequences to same length in a batch."""
    chars, tags = zip(*batch)
    
    # Pad sequences (padding value = 0)
    chars_padded = pad_sequence(chars, batch_first=True, padding_value=0)
    tags_padded = pad_sequence(tags, batch_first=True, padding_value=0)
    
    return chars_padded, tags_padded

def build_vocab(filepaths, min_freq=2):
    """Build character vocabulary from training files."""
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

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_correct = 0
    total_count = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    for chars, tags in progress_bar:
        chars, tags = chars.to(device), tags.to(device)
        
        # Create padding mask
        pad_mask = (chars == 0)
        
        # Forward pass
        outputs = model(chars, src_mask=pad_mask)  # [batch, seq_len, num_classes]
        
        # Calculate loss (ignore padding)
        loss = criterion(outputs.view(-1, outputs.size(-1)), tags.view(-1))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        mask = tags != 0  # Non-padding positions
        predictions = outputs.argmax(dim=-1)
        total_correct += ((predictions == tags) & mask).sum().item()
        total_count += mask.sum().item()
        
        progress_bar.set_postfix({'loss': loss.item(), 'acc': total_correct / total_count if total_count > 0 else 0})
    
    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_count if total_count > 0 else 0
    return avg_loss, accuracy

def evaluate(model, dataloader, criterion, device):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    total_correct = 0
    total_count = 0
    
    with torch.no_grad():
        for chars, tags in tqdm(dataloader, desc="Evaluating"):
            chars, tags = chars.to(device), tags.to(device)
            pad_mask = (chars == 0)
            
            outputs = model(chars, src_mask=pad_mask)
            loss = criterion(outputs.view(-1, outputs.size(-1)), tags.view(-1))
            
            total_loss += loss.item()
            mask = tags != 0
            predictions = outputs.argmax(dim=-1)
            total_correct += ((predictions == tags) & mask).sum().item()
            total_count += mask.sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_count if total_count > 0 else 0
    return avg_loss, accuracy

class TransformerTokenizer:
    def __init__(self, model, char2idx, idx2tag, device):
        self.model = model
        self.char2idx = char2idx
        self.idx2tag = idx2tag
        self.device = device
        self.model.eval()
    
    def tokenize(self, text, max_len=512):
        """Tokenize a single sentence."""
        # Convert characters to indices
        char_ids = [self.char2idx.get(c, self.char2idx['<UNK>']) for c in text]
        
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

def evaluate_segmentation(model, gold_filepath, test_filepath, char2idx, idx2tag, device):
    """Evaluate word segmentation performance."""
    tokenizer = TransformerTokenizer(model, char2idx, idx2tag, device)
    
    # Read test and gold data
    with open(test_filepath, 'r', encoding='utf-8') as f:
        test_lines = [line.strip() for line in f if line.strip()]
    
    with open(gold_filepath, 'r', encoding='utf-8') as f:
        gold_lines = [line.strip() for line in f if line.strip()]
    
    min_len = min(len(test_lines), len(gold_lines))
    test_lines = test_lines[:min_len]
    gold_lines = gold_lines[:min_len]
    
    total_pred_words = 0
    total_gold_words = 0
    total_correct_words = 0
    
    for test_line, gold_line in tqdm(zip(test_lines, gold_lines), total=len(test_lines), desc="Evaluating"):
        pred_words = tokenizer.tokenize(test_line)
        gold_words = gold_line.split()
        
        total_pred_words += len(pred_words)
        total_gold_words += len(gold_words)
        
        # Calculate correct words using position matching
        pred_positions = []
        pos = 0
        for word in pred_words:
            pred_positions.append((pos, pos + len(word), word))
            pos += len(word)
        
        gold_positions = []
        pos = 0
        for word in gold_words:
            gold_positions.append((pos, pos + len(word), word))
            pos += len(word)
        
        # Count matches
        for pred_pos in pred_positions:
            if pred_pos in gold_positions:
                total_correct_words += 1
    
    precision = total_correct_words / total_pred_words if total_pred_words > 0 else 0
    recall = total_correct_words / total_gold_words if total_gold_words > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {'precision': precision, 'recall': recall, 'f1': f1}

def plot_training_history(train_losses, train_accuracies, output_path='training_history.png'):
    """Plot training loss and accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot loss
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', linewidth=2, marker='o', markersize=6)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training Loss over Epochs', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(epochs)
    
    # Add value labels on points
    for i, loss in enumerate(train_losses):
        ax1.annotate(f'{loss:.4f}', (epochs[i], loss), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=9)
    
    # Plot accuracy
    ax2.plot(epochs, train_accuracies, 'g-', linewidth=2, marker='s', markersize=6)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Training Accuracy over Epochs', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(epochs)
    ax2.set_ylim([0, 1])
    
    # Add value labels on points
    for i, acc in enumerate(train_accuracies):
        ax2.annotate(f'{acc:.4f}', (epochs[i], acc), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Training history plot saved to {output_path}")
    plt.close()

if __name__ == "__main__":
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Hyperparameters
    BATCH_SIZE = 2048
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 10
    D_MODEL = 64
    NHEAD = 4
    NUM_LAYERS = 2
    DROPOUT = 0.1
    MAX_LEN = 256
    
    # Training files
    training_files = [
        '../icwb2-data/training/pku_training.utf8',
        '../icwb2-data/training/msr_training.utf8',
        '../icwb2-data/training/cityu_training.utf8',
        '../icwb2-data/training/as_training.utf8'
    ]
    existing_files = [f for f in training_files if os.path.exists(f)]
    print(f"Found {len(existing_files)} training files")
    
    # Build vocabulary
    print("Building vocabulary...")
    char2idx, tag2idx = build_vocab(existing_files, min_freq=2)
    idx2tag = {v: k for k, v in tag2idx.items()}
    vocab_size = len(char2idx)
    print(f"Vocabulary size: {vocab_size}")
    print(f"Tag vocabulary: {tag2idx}")
    
    # Save vocabulary
    with open('vocab.pkl', 'wb') as f:
        pickle.dump({'char2idx': char2idx, 'tag2idx': tag2idx, 'idx2tag': idx2tag}, f)
    print("Vocabulary saved to vocab.pkl")
    
    # Create datasets
    print("\nLoading training data...")
    train_datasets = []
    for filepath in existing_files:
        dataset = ChineseSegDataset(filepath, char2idx, tag2idx, max_len=MAX_LEN)
        train_datasets.append(dataset)
        print(f"  {filepath}: {len(dataset)} samples")
    
    # Combine all training data
    from torch.utils.data import ConcatDataset
    train_dataset = ConcatDataset(train_datasets)
    print(f"Total training samples: {len(train_dataset)}")
    
    # Create dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # Initialize model
    print("\nInitializing model...")
    model = TransformerSegModel(
        vocab_size=vocab_size,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        num_classes=len(tag2idx),
        dropout=DROPOUT
    ).to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model trainable parameters: {num_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    # Training loop
    print("\nStarting training...")
    best_loss = float('inf')
    train_losses = []
    train_accuracies = []
    
    for epoch in range(NUM_EPOCHS):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        print(f"{'='*50}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
        
        # Update learning rate
        scheduler.step(train_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning Rate: {current_lr}")
        
        # Save best model
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
            }, 'transformer_seg_best.pth')
            print(f"✓ Best model saved (loss: {best_loss:.4f})")
    
    # Plot training history
    print("\n" + "="*50)
    print("Generating training history plots...")
    print("="*50)
    plot_training_history(train_losses, train_accuracies, 'training_history.png')
     
    # Test on example sentences
    print("\n" + "="*50)
    print("Testing on example sentences")
    print("="*50)
    
    # Load best model
    checkpoint = torch.load('transformer_seg_best.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    tokenizer = TransformerTokenizer(model, char2idx, idx2tag, device)
    test_sentences = [
        "今天天气不错",
        "中国人民站起来了",
        "机器学习是人工智能的重要分支",
        "自然语言处理技术发展迅速"
    ]
    
    for sentence in test_sentences:
        words = tokenizer.tokenize(sentence)
        print(f"{sentence} -> {' / '.join(words)}")
    
    # Evaluate on test sets
    print("\n" + "="*50)
    print("Evaluating on test datasets")
    print("="*50)
    
    test_datasets = [
        ('PKU', '../icwb2-data/gold/pku_test_gold.utf8', '../icwb2-data/testing/pku_test.utf8'),
        ('MSR', '../icwb2-data/gold/msr_test_gold.utf8', '../icwb2-data/testing/msr_test.utf8'),
        ('CITYU', '../icwb2-data/gold/cityu_test_gold.utf8', '../icwb2-data/testing/cityu_test.utf8'),
        ('AS', '../icwb2-data/gold/as_testing_gold.utf8', '../icwb2-data/testing/as_test.utf8')
    ]
    
    all_results = {}
    for name, gold_path, test_path in test_datasets:
        if os.path.exists(gold_path) and os.path.exists(test_path):
            print(f"\nEvaluating {name}...")
            try:
                results = evaluate_segmentation(model, gold_path, test_path, char2idx, idx2tag, device)
                all_results[name] = results
                print(f"{name} - Precision: {results['precision']:.4f}, Recall: {results['recall']:.4f}, F1: {results['f1']:.4f}")
            except Exception as e:
                print(f"Error evaluating {name}: {e}")
    
    # Print summary
    if all_results:
        print("\n" + "="*50)
        print("Summary of All Datasets")
        print("="*50)
        print(f"{'Dataset':<10} {'Precision':<12} {'Recall':<12} {'F1':<12}")
        print("-" * 50)
        for name, results in all_results.items():
            print(f"{name:<10} {results['precision']:<12.4f} {results['recall']:<12.4f} {results['f1']:<12.4f}")
        
        avg_p = sum(r['precision'] for r in all_results.values()) / len(all_results)
        avg_r = sum(r['recall'] for r in all_results.values()) / len(all_results)
        avg_f1 = sum(r['f1'] for r in all_results.values()) / len(all_results)
        print("-" * 50)
        print(f"{'Average':<10} {avg_p:<12.4f} {avg_r:<12.4f} {avg_f1:<12.4f}")