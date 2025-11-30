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
import pickle
import sys
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List

import torch
import thulac

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DATASETS, SAVED_MODELS_DIR
from utils import BaseTokenizer, evaluate_model_on_all_datasets


# =============================================================================
# Segmenter Interface & Wrappers
# =============================================================================

class SegmenterInterface(ABC):
    """Abstract interface that all segmenters must implement."""
    
    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        """Segment text into words."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the model name."""
        pass


class ClassicalSegmenter(SegmenterInterface):
    """Wrapper for classical models (FMM, BMM, BiMM) - they already have tokenize()."""
    
    def __init__(self, model, model_name: str):
        self._model = model
        self._name = model_name
    
    def tokenize(self, text: str) -> List[str]:
        return self._model.tokenize(text)
    
    @property
    def name(self) -> str:
        return self._name


class HMMSegmenter(SegmenterInterface):
    """Wrapper for HMM model."""
    
    def __init__(self, model):
        self._model = model
    
    def tokenize(self, text: str) -> List[str]:
        return self._model.tokenize(text)
    
    @property
    def name(self) -> str:
        return "HMM"


class NeuralSegmenter(SegmenterInterface):
    """Wrapper for neural models (LSTM, RNN, Transformer)."""
    
    def __init__(self, model, char2idx: Dict, idx2tag: Dict, device, model_name: str):
        self._tokenizer = BaseTokenizer(model, char2idx, idx2tag, device)
        self._name = model_name
    
    def tokenize(self, text: str) -> List[str]:
        return self._tokenizer.tokenize(text)
    
    @property
    def name(self) -> str:
        return self._name


class ThulacSegmenter(SegmenterInterface):
    """Wrapper for THULAC library."""
    
    def __init__(self, model):
        self._model = model
    
    def tokenize(self, text: str) -> List[str]:
        result = self._model.cut(text)
        return [word if isinstance(word, str) else word[0] for word in result]
    
    @property
    def name(self) -> str:
        return "THULAC"


# =============================================================================
# Model Loaders
# =============================================================================

@dataclass
class LoadedModel:
    """Container for loaded model data."""
    segmenter: SegmenterInterface
    device: torch.device


def _get_device() -> torch.device:
    """Get the appropriate torch device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _get_training_files() -> List[str]:
    """Get list of existing training files for dictionary-based models."""
    training_files = [dataset['train'] for dataset in DATASETS.values()]
    return [f for f in training_files if os.path.exists(f)]


def load_fmm() -> LoadedModel:
    """Load Forward Maximum Matching model."""
    from models.classical import FMM
    
    existing_files = _get_training_files()
    print(f"Loading dictionary for FMM from {len(existing_files)} files...")
    
    model = FMM()
    model.load_dict(existing_files)
    print(f"Dictionary loaded with {len(model.vocab)} words for FMM")
    
    return LoadedModel(
        segmenter=ClassicalSegmenter(model, "FMM"),
        device=_get_device()
    )


def load_bmm() -> LoadedModel:
    """Load Backward Maximum Matching model."""
    from models.classical import BMM
    
    existing_files = _get_training_files()
    print(f"Loading dictionary for BMM from {len(existing_files)} files...")
    
    model = BMM()
    model.load_dict(existing_files)
    print(f"Dictionary loaded with {len(model.vocab)} words for BMM")
    
    return LoadedModel(
        segmenter=ClassicalSegmenter(model, "BMM"),
        device=_get_device()
    )


def load_bimm() -> LoadedModel:
    """Load Bidirectional Maximum Matching model."""
    from models.classical import BiMM
    
    existing_files = _get_training_files()
    print(f"Loading dictionary for BiMM from {len(existing_files)} files...")
    
    model = BiMM(dict_paths=existing_files)
    print(f"Dictionary loaded with {len(model.fmm.vocab)} words for BiMM")
    
    return LoadedModel(
        segmenter=ClassicalSegmenter(model, "BiMM"),
        device=_get_device()
    )


def load_hmm() -> LoadedModel:
    """Load HMM model."""
    from models.hmm_seg import HMMSeg
    
    model_path = f'{SAVED_MODELS_DIR}/hmm/hmm_model.pkl'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"HMM model not found at: {model_path}")
    
    model = HMMSeg()
    model.load_model(model_path)
    
    return LoadedModel(
        segmenter=HMMSegmenter(model),
        device=_get_device()
    )


def load_thulac() -> LoadedModel:
    """Load THULAC model."""
    model = thulac.thulac(seg_only=True)
    
    return LoadedModel(
        segmenter=ThulacSegmenter(model),
        device=_get_device()
    )


def _load_neural_model(model_type: str) -> LoadedModel:
    """Load a neural model (LSTM, RNN, or Transformer)."""
    from models.lstm_seg import LSTMSeg
    from models.rnn_seg import RNNSeg
    from models.transformer_seg import TransformerSeg
    from config import LSTM_CONFIG, RNN_CONFIG, TRANSFORMER_CONFIG
    
    device = _get_device()
    
    # Model configurations
    configs = {
        'lstm': {
            'class': LSTMSeg,
            'config': LSTM_CONFIG,
            'path': f'{SAVED_MODELS_DIR}/lstm/lstm_seg_best.pth',
            'init_fn': lambda cls, cfg, vocab_size, num_classes: cls(
                vocab_size=vocab_size,
                embedding_dim=cfg['embedding_dim'],
                hidden_dim=cfg['hidden_dim'],
                num_layers=cfg['num_layers'],
                num_classes=num_classes,
                dropout=cfg['dropout']
            )
        },
        'rnn': {
            'class': RNNSeg,
            'config': RNN_CONFIG,
            'path': f'{SAVED_MODELS_DIR}/rnn/rnn_seg_best.pth',
            'init_fn': lambda cls, cfg, vocab_size, num_classes: cls(
                vocab_size=vocab_size,
                d_model=cfg['d_model'],
                hidden_dim=cfg['hidden_dim'],
                num_layers=cfg['num_layers'],
                num_classes=num_classes
            )
        },
        'transformer': {
            'class': TransformerSeg,
            'config': TRANSFORMER_CONFIG,
            'path': f'{SAVED_MODELS_DIR}/transformer/transformer_seg_best.pth',
            'init_fn': lambda cls, cfg, vocab_size, num_classes: cls(
                vocab_size=vocab_size,
                d_model=cfg['d_model'],
                nhead=cfg['nhead'],
                num_layers=cfg['num_layers'],
                num_classes=num_classes,
                dropout=cfg['dropout']
            )
        }
    }
    
    model_cfg = configs[model_type]
    
    # Load vocabulary
    vocab_path = f'{SAVED_MODELS_DIR}/{model_type}/vocab.pkl'
    with open(vocab_path, 'rb') as f:
        vocab_data = pickle.load(f)
    char2idx = vocab_data['char2idx']
    tag2idx = vocab_data['tag2idx']
    idx2tag = vocab_data['idx2tag']
    
    # Initialize model
    model = model_cfg['init_fn'](
        model_cfg['class'],
        model_cfg['config'],
        len(char2idx),
        len(tag2idx)
    ).to(device)
    
    # Load trained weights
    checkpoint = torch.load(model_cfg['path'], map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    
    return LoadedModel(
        segmenter=NeuralSegmenter(model, char2idx, idx2tag, device, model_type.upper()),
        device=device
    )


def load_lstm() -> LoadedModel:
    """Load LSTM model."""
    return _load_neural_model('lstm')


def load_rnn() -> LoadedModel:
    """Load RNN model."""
    return _load_neural_model('rnn')


def load_transformer() -> LoadedModel:
    """Load Transformer model."""
    return _load_neural_model('transformer')


# =============================================================================
# Model Registry
# =============================================================================

# Registry mapping model names to their loader functions
MODEL_REGISTRY = {
    'fmm': load_fmm,
    'bmm': load_bmm,
    'bimm': load_bimm,
    'hmm': load_hmm,
    'thulac': load_thulac,
    'lstm': load_lstm,
    'rnn': load_rnn,
    'transformer': load_transformer,
}

SUPPORTED_MODELS = list(MODEL_REGISTRY.keys())


def load_model(model_type: str) -> LoadedModel:
    """
    Load a model by type using the registry.
    
    Args:
        model_type: One of the supported model types
        
    Returns:
        LoadedModel containing the segmenter and device
        
    Raises:
        ValueError: If model_type is not supported
    """
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}. Supported: {SUPPORTED_MODELS}")
    
    return MODEL_REGISTRY[model_type]()


# =============================================================================
# Evaluation Functions
# =============================================================================

TEST_SENTENCES = [
    "研究生命起源",
    "机器学习是人工智能的重要分支",
    "自然语言处理技术发展迅速"
]


def test_example_sentences(segmenter: SegmenterInterface) -> None:
    """Test model on example sentences."""
    print(f"\n{'='*60}")
    print(f"Testing {segmenter.name} on Example Sentences")
    print(f"{'='*60}")
    
    for sentence in TEST_SENTENCES:
        words = segmenter.tokenize(sentence)
        print(f"{sentence} -> {' / '.join(words)}")


def evaluate_model(model_type: str) -> None:
    """
    Evaluate a single model.
    
    Args:
        model_type: The type of model to evaluate
    """
    print(f"\nEvaluating {model_type.upper()} model...")
    
    # Load model (unified interface)
    loaded = load_model(model_type)
    
    # Test on example sentences
    test_example_sentences(loaded.segmenter)
    
    # Evaluate on all test datasets
    # The segmenter already has .tokenize() method
    evaluate_model_on_all_datasets(
        loaded.segmenter, 
        DATASETS, 
        f"{model_type.upper()}_segmentation"
    )


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained word segmentation models')
    parser.add_argument(
        '--model', 
        type=str, 
        required=True,
        choices=SUPPORTED_MODELS + ['all'],
        help='Model type to evaluate'
    )
    args = parser.parse_args()
    
    # Determine which models to evaluate
    models_to_eval = SUPPORTED_MODELS if args.model == 'all' else [args.model]
    
    for model_type in models_to_eval:
        try:
            evaluate_model(model_type)
        except Exception as e:
            print(f"Error evaluating {model_type}: {e}")
            continue


if __name__ == "__main__":
    main()
