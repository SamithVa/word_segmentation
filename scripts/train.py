#!/usr/bin/env python3
"""Training script for Chinese word segmentation models."""

import argparse
import os
import sys
import pickle

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    DATASETS, SAVED_MODELS_DIR, TRAINING_CONFIG,
    RNN_CONFIG, LSTM_CONFIG, TRANSFORMER_CONFIG,
    HMM_SMOOTHING, HMM_USE_PREPROCESS
)


def get_training_files():
    return [d['train'] for d in DATASETS.values() if os.path.exists(d['train'])]


def train_hmm():
    from models.hmm_seg import HMMSeg
    model_path = f'{SAVED_MODELS_DIR}/hmm/hmm_model.pkl'
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    hmm = HMMSeg(smoothing=HMM_SMOOTHING, use_preprocess=HMM_USE_PREPROCESS)
    hmm.train(get_training_files())
    hmm.save_model(model_path)
    print(f"HMM model saved to {model_path}")


def train_neural(model_type):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, ConcatDataset
    from utils import ChineseSegDataset, build_vocab, collate_fn, train_model
    from models.rnn_seg import RNNSeg
    from models.lstm_seg import LSTMSeg
    from models.transformer_seg import TransformerSeg

    configs = {'rnn': (RNNSeg, RNN_CONFIG), 'lstm': (LSTMSeg, LSTM_CONFIG), 'transformer': (TransformerSeg, TRANSFORMER_CONFIG)}
    model_class, config = configs[model_type]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training {model_type.upper()} on {device}")

    # Paths
    model_path = f'{SAVED_MODELS_DIR}/{model_type}/{model_type}_seg_best.pth'
    vocab_path = f'{SAVED_MODELS_DIR}/{model_type}/vocab.pkl'
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Build vocab
    files = get_training_files()
    char2idx, tag2idx = build_vocab(files, min_freq=2)
    with open(vocab_path, 'wb') as f:
        pickle.dump({'char2idx': char2idx, 'tag2idx': tag2idx, 'idx2tag': {v: k for k, v in tag2idx.items()}}, f)

    # Dataset
    full_dataset = ConcatDataset([ChineseSegDataset(f, char2idx, tag2idx, max_len=TRAINING_CONFIG['max_len']) for f in files])
    val_size = int(len(full_dataset) * TRAINING_CONFIG.get('val_split', 0.1))
    train_set, val_set = torch.utils.data.random_split(full_dataset, [len(full_dataset) - val_size, val_size])
    
    train_loader = DataLoader(train_set, batch_size=TRAINING_CONFIG['batch_size'], shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=TRAINING_CONFIG['batch_size'], collate_fn=collate_fn)

    # Model
    common = {'vocab_size': len(char2idx), 'num_classes': len(tag2idx)}
    if model_type == 'transformer':
        model = model_class(**common, **config)
    elif model_type == 'rnn':
        model = model_class(**common, d_model=config['d_model'], hidden_dim=config['hidden_dim'], num_layers=config['num_layers'])
    else:
        model = model_class(**common, embedding_dim=config['embedding_dim'], hidden_dim=config['hidden_dim'], 
                           num_layers=config['num_layers'], dropout=config['dropout'])
    
    model.to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Training
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=TRAINING_CONFIG['learning_rate'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, TRAINING_CONFIG['scheduler_step_size'], TRAINING_CONFIG['scheduler_gamma'])

    # Wandb
    use_wandb = TRAINING_CONFIG.get('use_wandb', False)
    if use_wandb:
        try:
            import wandb
            wandb.init(project=TRAINING_CONFIG.get('wandb_project', 'cws'), name=model_type, config={'model': model_type, **config})
        except ImportError:
            use_wandb = False

    train_model(model, train_loader, criterion, optimizer, scheduler, device,
                TRAINING_CONFIG['epochs'], model_path, val_loader, use_wandb)

    if use_wandb:
        import wandb
        wandb.finish()
    
    print(f"\n{model_type.upper()} saved to {model_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, choices=['hmm', 'rnn', 'lstm', 'transformer'])
    args = parser.parse_args()

    {'hmm': train_hmm, 'rnn': lambda: train_neural('rnn'), 
     'lstm': lambda: train_neural('lstm'), 'transformer': lambda: train_neural('transformer')}[args.model]()


if __name__ == "__main__":
    main()
