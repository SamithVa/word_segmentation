"""Models for Chinese Word Segmentation"""

# Core models (always available)
from .hmm_seg import HMMSeg as HMMTokenizer
from .classical import FMM, BMM, BiMM

# Optional neural models (require torch)
try:
    from .rnn_seg import RNNSeg as RNNTokenizer
    from .lstm_seg import LSTMSeg as LSTMTokenizer
    from .transformer_seg import TransformerSeg as TransformerTokenizer
    NEURAL_MODELS = True
except ImportError:
    NEURAL_MODELS = False
    RNNTokenizer = None
    LSTMTokenizer = None
    TransformerTokenizer = None

__all__ = [
    'HMMTokenizer',
    'FMM',
    'BMM',
    'BiMM',
    'RNNTokenizer',
    'LSTMTokenizer',
    'TransformerTokenizer',
    'NEURAL_MODELS',
]