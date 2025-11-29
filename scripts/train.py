#!/usr/bin/env python3
"""
Unified training script for Chinese word segmentation models.

Supported models: HMM, CRF, RNN, LSTM, Transformer

Usage:
    python scripts/train.py --model lstm
    python scripts/train.py --model hmm
"""

import argparse
import os
import sys
import pickle

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    DATASETS, SAVED_MODELS_DIR, TRAINING_CONFIG,
    RNN_CONFIG, LSTM_CONFIG, TRANSFORMER_CONFIG,
    HMM_SMOOTHING, HMM_USE_PREPROCESS
)

# Test sentences for validation
TEST_SENTENCES = [
    "中国的经济发展迅速",
    "北京是中国的首都",
    "我爱北京天安门"
]


def get_training_files():
    """Get list of existing training files."""
    training_files = [dataset['train'] for dataset in DATASETS.values()]
    existing_files = [f for f in training_files if os.path.exists(f)]
    print(f"Found {len(existing_files)} training files")
    return existing_files


def print_header(model_name):
    """Print training header."""
    print(f"\n{'='*50}")
    print(f"Training {model_name} Model")
    print(f"{'='*50}")


def test_model(model, model_name):
    """Test model with sample sentences."""
    print("\nTest predictions:")
    for sentence in TEST_SENTENCES:
        result = model.tokenize(sentence)
        print(f"  Input:  {sentence}")
        print(f"  Output: {result}")


def print_completion(model_name, model_path):
    """Print completion message."""
    print(f"\n✅ {model_name} training completed!")
    print(f"📁 Model saved to: {model_path}")
    print(f"\n💡 To evaluate: python scripts/eval.py --model {model_name.lower()}")


def train_hmm():
    """Train HMM model."""
    from models.hmm_seg import HMMSeg
    
    print_header("HMM")
    existing_files = get_training_files()

    model_path = f'{SAVED_MODELS_DIR}/hmm/hmm_model.pkl'
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    hmm = HMMSeg(smoothing=HMM_SMOOTHING, use_preprocess=HMM_USE_PREPROCESS)
    print("Training HMM model...")
    hmm.train(existing_files)
    hmm.save_model(model_path)

    test_model(hmm, "HMM")
    print_completion("HMM", model_path)

def train_crf():
    """Train CRF model."""
    try:
        import sklearn_crfsuite
    except ImportError:
        print("❌ Error: sklearn-crfsuite not installed!")
        print("Please install it with: pip install sklearn-crfsuite")
        sys.exit(1)

    from models.crf_seg import CRFSeg

    print_header("CRF")
    existing_files = get_training_files()

    model_path = f'{SAVED_MODELS_DIR}/crf/crf_model.pkl'
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    crf = CRFSeg()
    print("Training CRF model...")
    crf.train(existing_files)
    crf.save_model(model_path)

    test_model(crf, "CRF")
    print_completion("CRF", model_path)


def train_neural_model(model_type):
    """Train neural network models (RNN, LSTM, Transformer)."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, ConcatDataset

    from utils import ChineseSegDataset, build_vocab, collate_fn, train_model, plot_training_history
    from models.rnn_seg import RNNSeg
    from models.lstm_seg import LSTMSeg
    from models.transformer_seg import TransformerSeg

    print_header(model_type.upper())

    # Model configurations
    MODEL_CONFIGS = {
        'rnn': (RNNSeg, RNN_CONFIG),
        'lstm': (LSTMSeg, LSTM_CONFIG),
        'transformer': (TransformerSeg, TRANSFORMER_CONFIG)
    }

    if model_type not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model type: {model_type}")

    model_class, config = MODEL_CONFIGS[model_type]

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Get config values
    batch_size = TRAINING_CONFIG['batch_size']
    learning_rate = TRAINING_CONFIG['learning_rate']
    num_epochs = TRAINING_CONFIG['epochs']
    max_len = TRAINING_CONFIG['max_len']

    # Setup paths
    model_path = f'{SAVED_MODELS_DIR}/{model_type}/{model_type}_seg_best.pth'
    vocab_path = f'{SAVED_MODELS_DIR}/{model_type}/vocab.pkl'
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Get training files
    existing_files = get_training_files()

    # Build and save vocabulary
    print("\nBuilding vocabulary...")
    char2idx, tag2idx = build_vocab(existing_files, min_freq=2)
    idx2tag = {v: k for k, v in tag2idx.items()}
    vocab_size = len(char2idx)
    print(f"Vocabulary size: {vocab_size}")

    with open(vocab_path, 'wb') as f:
        pickle.dump({'char2idx': char2idx, 'tag2idx': tag2idx, 'idx2tag': idx2tag}, f)
    print(f"Vocabulary saved to {vocab_path}")

    # Create dataset and dataloader
    print("\nLoading training data...")
    train_datasets = [ChineseSegDataset(f, char2idx, tag2idx, max_len=max_len) for f in existing_files]
    train_dataset = ConcatDataset(train_datasets)
    print(f"Total training samples: {len(train_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0
    )

    # Initialize model based on type
    print(f"\nInitializing {model_type.upper()} model...")
    common_params = {'vocab_size': vocab_size, 'num_classes': len(tag2idx)}

    if model_type == 'transformer':
        model = model_class(
            **common_params,
            d_model=config['d_model'],
            nhead=config['nhead'],
            num_layers=config['num_layers'],
            dropout=config['dropout']
        )
    elif model_type == 'rnn':
        model = model_class(
            **common_params,
            d_model=config['d_model'],
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers']
        )
    else:  # lstm
        model = model_class(
            **common_params,
            embedding_dim=config['embedding_dim'],
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            dropout=config['dropout']
        )

    model = model.to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {param_count:,}")

    # Training setup
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=TRAINING_CONFIG['scheduler_step_size'],
        gamma=TRAINING_CONFIG['scheduler_gamma']
    )

    # Train model
    print("\nStarting training...")
    train_losses, train_accuracies = train_model(
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=num_epochs,
        model_path=model_path
    )

    # Plot training history
    plot_training_history(train_losses, train_accuracies, f'{model_type}_training_history.png', model_type.upper())

    # Print summary
    best_loss = min(train_losses) if train_losses else float('inf')
    final_acc = train_accuracies[-1] if train_accuracies else 0.0

    print(f"\n{'='*50}")
    print(f"✅ {model_type.upper()} training completed!")
    print(f"📁 Model: {model_path}")
    print(f"📊 Vocabulary: {vocab_path}")
    print(f"📈 Best loss: {best_loss:.4f}")
    print(f"🎯 Final accuracy: {final_acc:.4f}")
    print(f"{'='*50}")
    print(f"\n💡 To evaluate: python scripts/eval.py --model {model_type}")


def main():
    parser = argparse.ArgumentParser(description='Train Chinese word segmentation model')
    parser.add_argument(
        '--model', type=str, required=True,
        choices=['hmm', 'rnn', 'lstm', 'transformer', 'crf', 'fmm', 'bmm', 'bimm'],
        help='Model type to train'
    )
    args = parser.parse_args()

    # Dictionary-based models don't need training
    if args.model in ['fmm', 'bmm', 'bimm']:
        print(f"ℹ️  {args.model.upper()} is a dictionary-based model and does not require training.")
        print(f"💡 You can directly evaluate it: python scripts/eval.py --model {args.model}")
        return

    # Map model types to training functions
    trainers = {
        'hmm': train_hmm,
        'crf': train_crf,
        'rnn': lambda: train_neural_model('rnn'),
        'lstm': lambda: train_neural_model('lstm'),
        'transformer': lambda: train_neural_model('transformer'),
    }

    try:
        trainers[args.model]()
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()