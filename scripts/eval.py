#!/usr/bin/env python3
"""
Evaluation script for trained word segmentation models.

Usage:
    python eval.py --model lstm
    python eval.py --model rnn
    python eval.py --model transformer
    python eval.py --model hmm
    python eval.py --model all
"""

import argparse
import torch
import pickle
import sys
import os
import thulac


# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DATASETS, SAVED_MODELS_DIR
from utils import BaseTokenizer, evaluate_model_on_all_datasets


def load_classical_model(model_type):
    """Load classical models (FMM, BMM, BiMM)."""
    from models.classical import FMM, BMM, BiMM
    from config import DATASETS

    # Get training files for dictionary
    training_files = [dataset['train'] for dataset in DATASETS.values()]
    existing_files = [f for f in training_files if os.path.exists(f)]
    print(f"Loading dictionary for {model_type.upper()} from {len(existing_files)} files...")

    # Initialize and load dictionary
    if model_type == 'bimm':
        model = BiMM(dict_paths=existing_files)
        vocab_size = len(model.fmm.vocab)  # BiMM contains FMM and BMM models
    else:
        model = FMM() if model_type == 'fmm' else BMM()
        model.load_dict(existing_files)
        vocab_size = len(model.vocab)

    print(f"Dictionary loaded with {vocab_size} words for {model_type.upper()}")
    return model, {}, {}


def load_neural_model(model_type, device):
    """Load neural models (LSTM, RNN, Transformer)."""
    from models.lstm_seg import LSTMSeg
    from models.rnn_seg import RNNSeg
    from models.transformer_seg import TransformerSeg
    from config import LSTM_CONFIG, RNN_CONFIG, TRANSFORMER_CONFIG, TRAINING_CONFIG

    # Define model configuration mapping
    model_configs = {
        'lstm': (LSTMSeg, LSTM_CONFIG, f'{SAVED_MODELS_DIR}/lstm/lstm_seg_best.pth'),
        'rnn': (RNNSeg, RNN_CONFIG, f'{SAVED_MODELS_DIR}/rnn/rnn_seg_best.pth'),
        'transformer': (TransformerSeg, TRANSFORMER_CONFIG, f'{SAVED_MODELS_DIR}/transformer/transformer_seg_best.pth')
    }

    model_class, config, model_path = model_configs[model_type]

    # Load vocabulary
    vocab_path = f'{SAVED_MODELS_DIR}/{model_type}/vocab.pkl'
    with open(vocab_path, 'rb') as f:
        vocab_data = pickle.load(f)
    char2idx, tag2idx, idx2tag = vocab_data['char2idx'], vocab_data['tag2idx'], vocab_data['idx2tag']

    # Initialize model based on type

    # thulac (library for Chinese word segementation)

    if model_type == 'lstm':
        model = model_class(
            vocab_size=len(char2idx),
            embedding_dim=config['embedding_dim'],
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            num_classes=len(tag2idx),
            dropout=config['dropout']
        ).to(device)
    elif model_type == 'rnn':
        model = model_class(
            vocab_size=len(char2idx),
            d_model=config['d_model'],
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            num_classes=len(tag2idx)
        ).to(device)
    else:  # transformer
        model = model_class(
            vocab_size=len(char2idx),
            d_model=config['d_model'],
            nhead=config['nhead'],
            num_layers=config['num_layers'],
            num_classes=len(tag2idx),
            dropout=config['dropout'],
        ).to(device)

    # Load trained weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, char2idx, tag2idx, idx2tag


def load_hmm_model():
    """Load HMM model."""
    from models.hmm_seg import HMMSeg

    model_path = f'{SAVED_MODELS_DIR}/hmm/hmm_model.pkl'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"HMM model not found at: {model_path}")

    # Load model and create mappings
    model = HMMSeg()
    model.load_model(model_path)

    tag2idx = model.state_map
    idx2tag = {v: k for k, v in tag2idx.items()}
    all_chars = list(model.B.keys())
    char2idx = {char: idx for idx, char in enumerate(all_chars)}

    return model, char2idx, tag2idx, idx2tag


def load_thulac_model():
    """Load THULAC model for Chinese word segmentation."""
    
    # Initialize THULAC model (default mode: word segmentation)
    # THULAC API uses thulac.Thulac() class
    model = thulac.thulac(seg_only=True)
    
    # THULAC doesn't use vocabulary mappings like neural models
    # Return empty mappings for compatibility
    return model, {}, {}, {}


def load_model_and_vocab(model_type):
    """Load trained model and vocabulary."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_type in ['fmm', 'bmm', 'bimm']:
        model, char2idx, tag2idx = load_classical_model(model_type)
        idx2tag = {}  # Not used for classical models
        return model, char2idx, idx2tag, device
    elif model_type == 'hmm':
        model, char2idx, tag2idx, idx2tag = load_hmm_model()
        return model, char2idx, idx2tag, device
    elif model_type == 'thulac':
        model, char2idx, tag2idx, idx2tag = load_thulac_model()
        return model, char2idx, idx2tag, device
    else:
        model, char2idx, tag2idx, idx2tag = load_neural_model(model_type, device)
        return model, char2idx, idx2tag, device


def test_example_sentences(model, char2idx, idx2tag, device, model_type):
    """Test model on example sentences."""
    print(f"\n{'='*60}")
    print(f"Testing {model_type.upper()} on Example Sentences")
    print(f"{'='*60}")

    test_sentences = [
        "研究生命起源",
        "机器学习是人工智能的重要分支",
        "自然语言处理技术发展迅速"
    ]

    if model_type in ['fmm', 'bmm', 'bimm', 'hmm', 'thulac']:
        # Classical, HMM, and THULAC models use their native tokenize method
        for sentence in test_sentences:
            if model_type == 'thulac':
                # THULAC returns list of (word, tag) tuples when seg_only=False
                # or list of words when seg_only=True
                result = model.cut(sentence)
                words = [word if isinstance(word, str) else word[0] for word in result]
            else:
                words = model.tokenize(sentence)
            print(f"{sentence} -> {' / '.join(words)}")
    else:
        # Neural models use BaseTokenizer
        tokenizer = BaseTokenizer(model, char2idx, idx2tag, device)
        for sentence in test_sentences:
            words = tokenizer.tokenize(sentence)
            print(f"{sentence} -> {' / '.join(words)}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained word segmentation models')
    parser.add_argument('--model', type=str, required=True,
                       choices=['lstm', 'rnn', 'transformer', 'hmm', 'fmm', 'bmm', 'bimm', 'thulac', 'all'],
                       help='Model type to evaluate')
    args = parser.parse_args()

    models = ['lstm', 'rnn', 'transformer', 'hmm', 'fmm', 'bmm', 'bimm', 'thulac'] if args.model == 'all' else [args.model]

    for model_type in models:
        try:
            print(f"\nEvaluating {model_type.upper()} model...")

            # Load model and vocabulary
            model, char2idx, idx2tag, device = load_model_and_vocab(model_type)

            # Test on example sentences
            test_example_sentences(model, char2idx, idx2tag, device, model_type)

            # Evaluate on all test datasets
            if model_type in ['fmm', 'bmm', 'bimm', 'hmm']:
                tokenizer = model
            elif model_type == 'thulac':
                # Create a wrapper for THULAC to match the expected interface
                class ThulacWrapper:
                    def __init__(self, thulac_model):
                        self.thulac_model = thulac_model
                    
                    def tokenize(self, text):
                        result = self.thulac_model.cut(text)
                        return [word if isinstance(word, str) else word[0] for word in result]
                
                tokenizer = ThulacWrapper(model)
            else:
                tokenizer = BaseTokenizer(model, char2idx, idx2tag, device)
            
            evaluate_model_on_all_datasets(tokenizer, DATASETS, f"{model_type.upper()}_segmentation")

        except Exception as e:
            print(f"Error evaluating {model_type}: {e}")
            continue


if __name__ == "__main__":
    main()