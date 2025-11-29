# Chinese Word Segmentation System (中文分词系统)

This project implements and compares various Chinese Word Segmentation (CWS) algorithms, ranging from traditional rule-based methods to statistical models and modern deep learning approaches. It was developed to meet requirements of an NLP course project.

## 📋 Project Requirements & Features

The project fulfills the following requirements:

1. **Traditional Matching Algorithms**: Implemented Forward Maximum Matching (FMM), Backward Maximum Matching (BMM), and Bidirectional Maximum Matching (BiMM).
2. **Statistical Machine Learning**: Implemented Hidden Markov Model (HMM) with Viterbi decoding.
3. **Deep Learning**: Implemented RNN, LSTM, Transformer, and CRF models using PyTorch.
4. **Performance Comparison**: Detailed evaluation and comparison of all algorithms on SIGHAN Bakeoff 2005 datasets (PKU, MSR, CITYU, AS).
5. **Interactive UI**: A user-friendly web interface built with Gradio to demonstrate and compare segmentation results in real-time.

## 🚀 Quick Start

### Training Models

**Important**: All model training is done through a single script: `scripts/train.py`.

#### Classical Dictionary-based Models (No dependencies)
```bash
# Train HMM model (fast, no dependencies)
python scripts/train.py --model hmm

# Train classical maximum matching methods (no dependencies)
python scripts/train.py --model fmm   # Forward Maximum Matching
python scripts/train.py --model bmm   # Backward Maximum Matching
python scripts/train.py --model bimm  # Bidirectional Maximum Matching
```

#### Machine Learning Models
```bash
# Train neural network models (requires PyTorch)
python scripts/train.py --model lstm
python scripts/train.py --model rnn
python scripts/train.py --model transformer

# Train CRF model (requires sklearn-crfsuite)
python scripts/train.py --model crf
```

### Evaluating Models

After training, evaluate models using:

```bash
python scripts/eval.py --model <model_type>
```

Available model types: `hmm`, `rnn`, `lstm`, `transformer`, `crf`, `fmm`, `bmm`, `bimm`, `all`

### Interactive Demo

Launch the Gradio web interface:

```bash
python ui_gradio.py
```

## 📂 Directory Structure

```
.
├── __init__.py                    # Package initialization
├── config.py                      # Configuration file with all settings
├── models/                        # Model implementations
│   ├── __init__.py
│   ├── base.py                    # Base classes
│   ├── hmm_seg.py                 # HMM implementation
│   ├── classical.py               # FMM, BMM, BiMM implementations
│   ├── rnn_seg.py                 # RNN model
│   ├── lstm_seg.py                # LSTM model
│   ├── transformer_seg.py         # Transformer model
│   └── crf_seg.py                 # CRF model
├── scripts/                       # Training and evaluation scripts
│   ├── __init__.py
│   ├── train.py                   # Unified training script for all models
│   └── eval.py                    # Updated evaluation script
├── utils/                         # Shared utilities
│   ├── __init__.py
│   ├── dataset.py                 # Dataset helper functions
│   ├── evaluation.py              # Segmentation metrics
│   ├── tokenizer.py               # Tokenizer utilities
│   ├── trainer.py                 # Training utilities
│   ├── visualization.py           # Visualization utilities
│   └── plot_analysis.py           # Plotting utilities
├── outputs/                       # Generated outputs
│   ├── saved_models/              # Trained model weights
│   │   ├── hmm/
│   │   ├── lstm/
│   │   ├── transformer/
│   │   ├── rnn/
│   │   └── crf/
│   └── results/                   # Evaluation results
├── icwb2-data/                    # ICWB2 dataset directory
├── ui_gradio.py                   # Gradio Web UI
└── requirements.txt               # Python dependencies
```

## 🎯 Model Details

### Classical Dictionary-based Models

#### 1. HMM (Hidden Markov Model)
- **Dependencies**: None (uses only numpy)
- **Speed**: Very fast training and inference
- **Performance**: Good baseline, works well on small datasets
- **Features**: BIES tagging, preprocessing for numbers/English

#### 2. FMM (Forward Maximum Matching)
- **Dependencies**: None
- **Speed**: Very fast, no training required
- **Method**: Greedy forward matching with longest words first
- **Best for**: Simple dictionary-based segmentation

#### 3. BMM (Backward Maximum Matching)
- **Dependencies**: None
- **Speed**: Very fast, no training required
- **Method**: Greedy backward matching with longest words first
- **Best for**: Complement to FMM, different segmentation patterns

#### 4. BiMM (Bidirectional Maximum Matching)
- **Dependencies**: None
- **Speed**: Very fast, no training required
- **Method**: Combines FMM and BMM results with rules
- **Best for**: More robust than single-direction methods

### Machine Learning Models

#### 5. RNN (Recurrent Neural Network)
- **Dependencies**: PyTorch
- **Architecture**: Simple RNN with embedding layer
- **Parameters**: 128-dim embeddings, 256-dim hidden states

#### 6. LSTM (Long Short-Term Memory)
- **Dependencies**: PyTorch
- **Architecture**: BiLSTM with embedding layer
- **Parameters**: 128-dim embeddings, 256-dim hidden states
- **Performance**: Typically better than simple RNN

#### 7. Transformer
- **Dependencies**: PyTorch
- **Architecture**: Multi-head self-attention
- **Parameters**: 128-dim model, 4 attention heads
- **Features**: Positional encoding, multi-layer attention

#### 8. CRF (Conditional Random Field)
- **Dependencies**: sklearn-crfsuite
- **Features**: Hand-crafted features, BMES tagging
- **Performance**: Strong baseline, good with feature engineering

## 📊 Updated Evaluation System

The evaluation system has been significantly improved:

- **New evaluation script**: `scripts/eval.py` replaces the old `scripts/evaluate_model.py`
- **Comprehensive support**: All models (including classical models) are now properly supported
- **Cleaner code**: Refactored with modular functions for better maintainability
- **Better error handling**: Improved exception handling for robust evaluation
- **Unified interface**: Consistent evaluation process for all model types

## ⚙️ Configuration

All model configurations are centralized in `config.py`:

```python
# Shared training parameters
TRAINING_CONFIG = {
    'batch_size': 512,
    'learning_rate': 0.001,
    'epochs': 10,
    'max_len': 256,
    # ...
}

# Model-specific parameters
SHARED_EMBEDDING_DIM = 128
SHARED_HIDDEN_DIM = 256
SHARED_NUM_LAYERS = 2
SHARED_DROPOUT = 0.3
```

## 📊 Output Structure

Trained models are saved to `outputs/saved_models/`:

```
outputs/
└── saved_models/
    ├── hmm/
    │   └── hmm_model.pkl
    ├── lstm/
    │   ├── bilstm_seg_best.pth     # Updated naming
    │   ├── vocab.pkl
    │   └── lstm_training_history.png
    ├── transformer/
    │   ├── transformer_seg_best.pth
    │   ├── transformer_seg_final.pth
    │   ├── vocab.pkl
    │   └── transformer_training_history.png
    └── ...
```

## 📝 Data Format

Training data should be in UTF-8 format with space-separated words:

```
中国 的 经济 发展 迅速
北京 是 中国 的 首都
```

Available datasets:
- PKU (pku_training.utf8)
- MSR (msr_training.utf8)
- CityU (cityu_training.utf8)
- AS (as_training.utf8)

## 💡 Updated Tips

1. **For quick testing**: Use HMM model - no dependencies needed
2. **For best performance**: Try Transformer or BiLSTM
3. **For feature engineering**: Use CRF with custom features
4. **GPU training**: Neural models automatically use GPU if available
5. **Batch size**: Adjust based on your GPU memory
6. **New evaluation**: Use `scripts/eval.py` for all model types including classical models
7. **Model comparison**: Run `python scripts/eval.py --model all` for comprehensive comparison

## 🔧 Troubleshooting

### PyTorch not installed
```bash
pip install torch torchvision
```

### sklearn-crfsuite not installed (for CRF)
```bash
pip install sklearn-crfsuite
```

### CUDA out of memory
Reduce `batch_size` in `config.py`:
```python
TRAINING_CONFIG = {
    'batch_size': 256,  # Reduced from 512
    # ...
}
```

## 🔧 Recent Improvements

### Code Structure & Maintainability
- **Refactored Evaluation Script**: The new `scripts/eval.py` is cleaner, more modular, and easier to maintain
- **Fixed Variable Conflicts**: Resolved issues with variable name conflicts in the main function
- **Improved Neural Model Loading**: Better handling of different parameter names for LSTM, RNN, and Transformer models
- **Enhanced Classical Model Support**: Proper handling of BiMM model which contains FMM and BMM components

### Consistency Improvements
- **Updated Training Script**: Now properly handles BiMM model by accessing vocab size from underlying FMM model
- **Consistent Naming**: Updated evaluation calls to use the new script name (`eval.py`)
- **Better Error Messages**: More informative error outputs during evaluation

### Architecture Benefits

- **Simplicity**: Clean, flat structure that's easy to navigate
- **Modularity**: Each model is self-contained in its own file
- **Reusability**: Shared utilities reduce code duplication
- **Consistency**: Standardized configuration through `config.py`
- **Unified Interface**: Single training script for all models
- **Improved Evaluation Process**: More robust and comprehensive evaluation across all model types